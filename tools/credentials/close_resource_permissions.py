"""Audit — and close — `Role.any()` grants on this platform's Appwrite resources.

THE EXPOSURE THIS WAS WRITTEN FOR — CLOSED 2026-08-14
-----------------------------------------------------
`unfao` and `production_forecasts` carried
`Permission.{read,create,update,delete}(Role.any())` with `documentSecurity: false`.
Collection-level grants govern when document security is off, and `Role.any()` includes
guests, so an unauthenticated caller holding only the project ID could read, rewrite and
delete FAO's delivery metadata and the production forecast index. Measured before the
fix: an anonymous `listDocuments` returned all 111 `unfao` rows and all 461
`production_forecasts` rows. Both now answer 401.

There was no obscurity barrier. The project ID is tracked on the public default branch of
views-appwrite (`docs/ADRs/platform/coordinate_registry.toml`), and views-pipeline-core's
public register already described the grants at C-292 — so the route was fully mapped in
the open.

`update` was the dangerous verb, not `delete`. `file_hash` is on 100% of rows and was
attacker-writable, so rewriting `fileId` and `file_hash` together repoints a record at
substituted content *and* fixes up the only integrity control. A deletion is loud; that
is not.

**This script stays as the regression guard.** Nothing else on the platform inspects a
resource's permission list — C-292's own wording is *"No test inspects the argument"* —
and the upstream default that produced the grants is unchanged. Run it after any
provisioning; a clean run prints "nothing to do" and exits 0.

WHY CLOSING IT BREAKS NOTHING
-----------------------------
`AuthMethod` is a single-member enum (`API_KEY`); session auth was deleted platform-wide
(þing-01 #274 / C-255, 2026-08-01); there is no JS or web client anywhere; FAO's shipped
notebooks call `faoapi.viewsforecasting.org` with `X-API-Key` and never import `appwrite`.
API keys bypass resource permissions entirely.

The decisive proof is already in this platform's history: the CRAF'd delivery reads and
writes `crafd`, which has `$permissions: []`, under the datastore key. It failed on a
schema `AttributeError`, never a 401.

So the target is `[]` — not a narrower role. Any role invented here would be decoration,
and `crafd` already demonstrated the empty list works: it has run at `permissions: []`
throughout while the delivery read and wrote it under the datastore key. **`crafd` was the
reference state, not the odd one out** — the asymmetry was a provenance artifact of
whichever code path created each collection. Confirmed after the fix: all three
collections read normally with the key (111 / 461 / 111, unchanged) and refuse anonymously.

The root cause is upstream: `views-pipeline-core .../modules/appwrite/provisioning.py`
passes `Role.any()` when creating a collection while the sibling `ensure_bucket` in the
same module defaults to `permissions=[]`. One command produces a locked bucket and an open
collection. This script cleans up what that already emitted; it does not fix it.

WHAT THIS TOUCHES, AND WHAT IT REFUSES TO
-----------------------------------------
Audits every collection AND bucket the registry declares. **Mutates collections only.**

Buckets are refused deliberately. `PUT /storage/buckets/{id}` resets `maximumFileSize`,
`allowedFileExtensions`, `compression`, `encryption` and `antivirus` when they are omitted
— a far larger read-modify-write blast radius than the collection endpoint's three fields.
All three buckets are already closed, so the risk buys nothing. If one ever shows a grant,
this reports it loudly and tells the operator to close it in the console.

USAGE
-----
    . tools/credentials/platform_env.sh && platform_env_load
    python tools/credentials/close_resource_permissions.py             # audit only
    python tools/credentials/close_resource_permissions.py --apply     # close them

Needs Python 3.11+ on PATH for the registry read (`tomllib`); base is 3.10.

Dry-run by default. `crafd` doubles as the control: it should report "already closed",
which is how we know the target state is observed rather than invented.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

TIMEOUT = 60

# (label, collection-id var, bucket-id var). Names come from the registry — the ONE source
# of coordinates (ADR-018) — so adding a partner here means adding it to the registry.
RESOURCES = (
    ("unfao", "APPWRITE_UNFAO_COLLECTION_ID", "APPWRITE_UNFAO_BUCKET_ID"),
    ("production_forecasts", "APPWRITE_PROD_FORECASTS_COLLECTION_ID",
     "APPWRITE_PROD_FORECASTS_BUCKET_ID"),
    ("crafd", "APPWRITE_CRAFD_COLLECTION_ID", "APPWRITE_CRAFD_BUCKET_ID"),
)

BASE_VARS = ("APPWRITE_ENDPOINT", "APPWRITE_DATASTORE_PROJECT_ID",
             "APPWRITE_DATASTORE_API_KEY", "APPWRITE_METADATA_DATABASE_ID")


def _count_query() -> str:
    """`limit(1)` as an Appwrite query string, URL-ENCODED.

    The JSON contains spaces and urllib refuses a path with control characters outright
    (`InvalidURL`), so the naive f-string form fails before any request is sent. Same
    encoding as `tools/liveness/appwrite_api.py::newest_first_query`.
    """
    return "queries[]=" + quote(json.dumps({"method": "limit", "values": [1]}))


def _call(method: str, url: str, headers: dict, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = Request(url, data=data, headers=headers, method=method)
    with urlopen(req, timeout=TIMEOUT) as response:
        raw = response.read()
        return json.loads(raw) if raw else {}


def _probe_anonymous(url: str, project_id: str) -> str:
    """What an unauthenticated caller holding only the project ID gets back.

    Deliberately does NOT reduce to a boolean. Appwrite answers a rejected key on the
    file-listing endpoint with HTTP 200 and `total: 0` rather than 401 (measured
    2026-08-02 against 1.9.5, and documented at
    `tools/liveness/appwrite_api.py::assert_bucket_reachable`) — a shape that renders a
    refusal as emptiness. Reporting the status and the count separately keeps that
    distinguishable instead of collapsing it into the answer we hoped for.
    """
    headers = {"X-Appwrite-Project": project_id}
    try:
        body = _call("GET", url, headers)
    except HTTPError as exc:
        return f"HTTP {exc.code} — refused"
    except URLError as exc:
        return f"unreachable: {exc.reason}"
    total = body.get("total")
    if total:
        return f"HTTP 200, total={total}  <-- READABLE BY ANYONE"
    return f"HTTP 200, total={total} (accepted the request but returned nothing)"


def _audit_collection(ep: str, db: str, coll: str, headers: dict, project_id: str) -> dict:
    state = _call("GET", f"{ep}/databases/{db}/collections/{coll}", headers)
    docs_url = f"{ep}/databases/{db}/collections/{coll}/documents"
    keyed = _call("GET", f"{docs_url}?{_count_query()}", headers)["total"]
    return {
        "id": coll,
        "name": state["name"],
        "permissions": state.get("$permissions", []),
        "documentSecurity": state.get("documentSecurity"),
        "enabled": state.get("enabled"),
        "keyed_total": keyed,
        "anonymous": _probe_anonymous(f"{docs_url}?{_count_query()}", project_id),
    }


def _close_collection(ep: str, db: str, headers: dict, before: dict) -> list:
    """Empty the permission list, preserving everything else the endpoint can reset.

    `PUT /databases/{db}/collections/{id}` takes `name` as REQUIRED and resets omitted
    optional parameters to their defaults. A naive PUT carrying only `permissions` would
    therefore rename the collection and silently flip `documentSecurity`. Read first, pass
    the rest back through unchanged, mutate one field.
    """
    body = {
        "name": before["name"],
        "permissions": [],
        "documentSecurity": before["documentSecurity"],
        "enabled": before["enabled"],
    }
    after = _call("PUT", f"{ep}/databases/{db}/collections/{before['id']}", headers, body)

    # The read-modify-write is the risky part, so verify it rather than trusting the 200.
    drift = [
        f"{field}: {before[field]!r} -> {after.get(key)!r}"
        for field, key in (("name", "name"),
                           ("documentSecurity", "documentSecurity"),
                           ("enabled", "enabled"))
        if after.get(key) != before[field]
    ]
    if after.get("$permissions"):
        drift.append(f"permissions not emptied: {after['$permissions']}")
    return drift


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--apply", action="store_true",
                        help="actually close the collections (default: audit only)")
    parser.add_argument("--only", choices=[label for label, _, _ in RESOURCES],
                        help="restrict to one resource")
    args = parser.parse_args()

    targets = [r for r in RESOURCES if args.only in (None, r[0])]
    need = list(BASE_VARS) + [v for _, coll_var, bucket_var in targets
                              for v in (coll_var, bucket_var)]
    if missing := [v for v in need if not os.getenv(v)]:
        print(f"FATAL: not exported: {missing}\n"
              f"  Run: . tools/credentials/platform_env.sh && platform_env_load\n"
              f"  (needs Python 3.11+ on PATH — base is 3.10 and tomllib is 3.11+)",
              file=sys.stderr)
        return 1

    ep = os.environ["APPWRITE_ENDPOINT"].rstrip("/")
    db = os.environ["APPWRITE_METADATA_DATABASE_ID"]
    project_id = os.environ["APPWRITE_DATASTORE_PROJECT_ID"]
    headers = {
        "X-Appwrite-Project": project_id,
        "X-Appwrite-Key": os.environ["APPWRITE_DATASTORE_API_KEY"],
        "Content-Type": "application/json",
    }

    print("MODE: apply" if args.apply else "MODE: audit only (pass --apply to close)")
    print()

    open_collections: list[dict] = []
    open_buckets: list[str] = []

    for label, coll_var, bucket_var in targets:
        state = _audit_collection(ep, db, os.environ[coll_var], headers, project_id)
        grants = state["permissions"]
        print(f"collection {label!r} ({state['id']})")
        print(f"    documents (with key) : {state['keyed_total']}")
        print(f"    documents (anonymous): {state['anonymous']}")
        print(f"    documentSecurity     : {state['documentSecurity']}")
        if grants:
            print(f"    permissions          : {grants}   <-- OPEN")
            open_collections.append(state)
        else:
            print("    permissions          : []   (already closed)")

        bucket = _call("GET", f"{ep}/storage/buckets/{os.environ[bucket_var]}", headers)
        bucket_grants = bucket.get("$permissions", [])
        print(f"    bucket {bucket['$id']}: permissions={bucket_grants or '[]'} "
              f"fileSecurity={bucket.get('fileSecurity')}")
        if bucket_grants:
            open_buckets.append(f"{label} bucket {bucket['$id']}: {bucket_grants}")
        print()

    if open_buckets:
        # Refused on purpose — see the module docstring. Reporting beats a wide PUT.
        print("BUCKETS CARRY GRANTS — close these in the console, not here:", file=sys.stderr)
        for line in open_buckets:
            print(f"    {line}", file=sys.stderr)
        print(file=sys.stderr)

    if not open_collections:
        print("Nothing to do: every collection audited already has an empty permission list.")
        return 1 if open_buckets else 0

    if not args.apply:
        print(f"{len(open_collections)} collection(s) would be closed to `permissions: []`, "
              f"leaving documentSecurity and enabled untouched. Re-run with --apply.")
        return 0

    failed = False
    for before in open_collections:
        print(f"closing {before['name']!r} ({before['id']}) ...")
        drift = _close_collection(ep, db, headers, before)
        if drift:
            failed = True
            print(f"  FATAL: the update changed more than permissions — {'; '.join(drift)}\n"
                  f"    RESTORE BY HAND: name={before['name']!r} "
                  f"documentSecurity={before['documentSecurity']} "
                  f"enabled={before['enabled']}", file=sys.stderr)
            continue
        after = _audit_collection(ep, db, before["id"], headers, project_id)
        print(f"  permissions          : {after['permissions'] or '[]'}")
        print(f"  documents (with key) : {after['keyed_total']}  "
              f"(was {before['keyed_total']})")
        print(f"  documents (anonymous): {after['anonymous']}")
        if after["keyed_total"] != before["keyed_total"]:
            failed = True
            print(f"  FATAL: document count moved {before['keyed_total']} -> "
                  f"{after['keyed_total']}. Nothing here touches documents.", file=sys.stderr)

    if failed or open_buckets:
        return 1
    print("\nClosed. Re-run without --apply to confirm the audit is clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
