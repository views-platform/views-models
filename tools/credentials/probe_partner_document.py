"""Prove a partner metadata collection accepts the document a delivery actually writes.

WHY, RATHER THAN JUST RUNNING THE DELIVERY
------------------------------------------
`_require_containers` (pipeline-core `modules/appwrite/file.py:1400-1446`) checks that the
collection EXISTS. It never calls `list_attributes`. So a collection whose schema does not
fit passes the preflight, and the refusal lands at `create_document` — which is step 10,
AFTER `create_file` has already put the bytes in the bucket.

**Every rejected field therefore costs one orphaned file per shard.** A 108-shard delivery
against a schema that is wrong in one attribute leaves 108 unreferenced files behind.

This writes ONE document with the exact payload the delivery sends, then deletes it. If
Appwrite accepts it, the delivery's document write cannot fail on schema. If it does not,
we learn that for the price of one document.

WHY NOT THE DISARMED DRY RUN
----------------------------
With `intent = paused`, `wire_upload_enabled` is false, `unfao.py:355` sets `store = None`,
and `sink.py:150` returns before `_upload`. A disarmed run makes ZERO Appwrite calls on the
write path — it cannot test a schema change.

THE PAYLOAD
-----------
Enumerated from the installed code, not guessed:
  file.py:2152-2158  fileId, filename, bucketId, uploaded_at, file_hash
  sink.py:161        loa, name, category
  crafd.py:48-52     type, targets, description

`uploaded_at` is the one genuinely untested field: `file.py:2157` writes
`datetime.now().isoformat()` — **naive, local clock, no timezone** — into an attribute
declared `datetime`. Whether Appwrite coerces it, and to what, is not determinable from the
source. That is exactly the class of per-field mismatch that produced the run-0
`description` overflow.

USAGE
-----
    python tools/credentials/probe_partner_document.py un_crafd

Read-mostly and self-cleaning: every document it creates is deleted in a `finally`, and it
verifies the collection is back to its starting count before reporting success.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

CONSUMERS = {
    "un_crafd": ("APPWRITE_CRAFD_COLLECTION_ID", "APPWRITE_CRAFD_BUCKET_ID"),
    "un_fao": ("APPWRITE_UNFAO_COLLECTION_ID", "APPWRITE_UNFAO_BUCKET_ID"),
}
TIMEOUT = 60


def _count_query() -> str:
    """`limit(1)` as an Appwrite query string, URL-ENCODED.

    The JSON contains spaces, and urllib refuses a path with control characters outright
    (`InvalidURL`), so the naive f-string form fails before any request is sent. Same
    encoding as `tools/liveness/appwrite_api.py::newest_first_query`; kept local because
    that module resolves its own credentials and this script takes them from the
    environment the launcher already exports.
    """
    return "queries[]=" + quote(json.dumps({"method": "limit", "values": [1]}))


def _call(method: str, url: str, headers: dict, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = Request(url, data=data, headers=headers, method=method)
    with urlopen(req, timeout=TIMEOUT) as r:
        raw = r.read()
        return json.loads(raw) if raw else {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("consumer", choices=sorted(CONSUMERS))
    args = ap.parse_args()

    coll_var, bucket_var = CONSUMERS[args.consumer]
    need = ["APPWRITE_ENDPOINT", "APPWRITE_DATASTORE_PROJECT_ID",
            "APPWRITE_DATASTORE_API_KEY", "APPWRITE_METADATA_DATABASE_ID",
            coll_var, bucket_var]
    if missing := [v for v in need if not os.getenv(v)]:
        print(f"FATAL: not exported: {missing}\n"
              f"  Run: . tools/credentials/platform_env.sh && platform_env_load\n"
              f"  (needs Python 3.11+ on PATH — base is 3.10 and tomllib is 3.11+)",
              file=sys.stderr)
        return 1

    ep = os.environ["APPWRITE_ENDPOINT"].rstrip("/")
    db = os.environ["APPWRITE_METADATA_DATABASE_ID"]
    coll = os.environ[coll_var]
    headers = {
        "X-Appwrite-Project": os.environ["APPWRITE_DATASTORE_PROJECT_ID"],
        "X-Appwrite-Key": os.environ["APPWRITE_DATASTORE_API_KEY"],
        "Content-Type": "application/json",
    }
    docs_url = f"{ep}/databases/{db}/collections/{coll}/documents"

    # The wire payload (11 keys) and the historical payload (12 — description is written
    # only by crafd.py:394, after the manifest, and is the run-0 failure field).
    wire = {
        "loa": "pgm",
        "name": args.consumer,
        "type": "sampled_forecast_shard",
        "targets": ["lr_ged_sb"],
        "category": "forecast",
        "fileId": "probe0000000000000000",
        "filename": "probe__lr_ged_sb__m000559.arrow.parquet",
        "bucketId": os.environ[bucket_var],
        "uploaded_at": datetime.now().isoformat(),   # NAIVE — deliberately, file.py:2157
        "file_hash": "0" * 64,                       # 64 hex, file_hash is size=64 exactly
    }
    historical = {
        **wire,
        "type": "model",
        "category": "historical",
        "filename": "probe_historical_dataset.parquet",
        "description": json.dumps({"probe": True, "note": "provenance blob stand-in"}),
    }

    before = _call("GET", f"{docs_url}?{_count_query()}", headers)["total"]
    print(f"collection {coll!r}: {before} documents before")

    created: list[str] = []
    failures: list[str] = []
    try:
        for label, payload in (("wire (11 keys)", wire), ("historical (12 keys)", historical)):
            try:
                doc = _call("POST", docs_url, headers,
                            {"documentId": "unique()", "data": payload})
                created.append(doc["$id"])
                echoed = doc.get("uploaded_at")
                print(f"  ACCEPTED  {label}")
                if label.startswith("wire"):
                    print(f"      uploaded_at sent : {payload['uploaded_at']}")
                    print(f"      uploaded_at back : {echoed}   <- the untested coercion")
                    print(f"      targets     back : {doc.get('targets')!r}")
            except HTTPError as e:
                detail = e.read().decode(errors="replace")[:400]
                failures.append(f"{label}: HTTP {e.code} {detail}")
                print(f"  REJECTED  {label}\n      HTTP {e.code} {detail}", file=sys.stderr)
    finally:
        # Self-cleaning: a probe that leaves rows behind corrupts the orphan count that is
        # the delivery's only real verification.
        for doc_id in created:
            try:
                _call("DELETE", f"{docs_url}/{doc_id}", headers)
                print(f"  cleaned up {doc_id}")
            except HTTPError as e:
                print(f"  WARNING: could not delete {doc_id}: HTTP {e.code} — "
                      f"DELETE IT BY HAND before delivering", file=sys.stderr)

    after = _call("GET", f"{docs_url}?{_count_query()}", headers)["total"]
    print(f"collection {coll!r}: {after} documents after")

    if after != before:
        print(f"\nFATAL: document count changed {before} -> {after}. Clean up before "
              f"delivering — a dirty collection breaks the orphan check.", file=sys.stderr)
        return 1
    if failures:
        print("\nFATAL: the schema does NOT accept the delivery payload. Do not run the "
              "delivery — each rejected field orphans one file per shard.", file=sys.stderr)
        return 1
    print("\nBoth payloads accepted and cleaned up. The document write cannot fail on schema.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
