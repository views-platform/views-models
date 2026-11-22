"""Read a published forecast BACK off the production shelf and verify it against its own manifests.

The publisher's last line is "Read it back from the shelf by that run_id before trusting it."
This is that step. Nothing else in the repository does it: `tools/audit/verify_wire.py` verifies a
LOCAL wire directory against a local manifest, which is a different artefact at a different stage.

READ-ONLY. It lists, reads metadata, and downloads. It never uploads, updates or deletes.

⛔ THE TRAP THIS TOOL EXISTS TO AVOID. `DatastoreModule.download_prediction` defaults to
`use_cache=True`. Measured 2026-11-13 while verifying chrome_box_forecasting_20261112_210150: it
returned `code=RETURNED_FROM_CACHE`, `from_cache=True` and 266 MB of the UPLOADER'S OWN cached copy
of what it had just uploaded. Hashing those bytes verifies the uploader's disk and says nothing
about what a partner will fetch — a false pass that looks identical to a real one. So every download
here passes `use_cache=False, validate_cache=False`, and a result that still comes back
`from_cache=True` is treated as a FAILURE rather than ignored. (ADR-025: a check that cannot see its
subject must refuse, not skip.)

TWO CHECKS, DELIBERATELY SEPARATE, because they fail differently:
  (a) every shard's server-recorded `file_hash` against the sha256 in the manifest the server holds.
      This is the server's own CLAIM about all of its shards. Cheap, complete.
  (b) a sample of shards fetched and re-hashed here. This is the actual BYTES. Expensive, partial.
Check (a) alone would pass if the server had stored a wrong hash alongside wrong bytes consistently,
which is why (b) is not optional and why a run with zero successful (b) samples does not verify.

⚠️ A manifest fetched back is NOT byte-identical to the stored one. The SDK parses a JSON payload
and pipeline-core re-serializes it in compact form (#310 / register C-217), which the download
announces on stderr. So this tool PARSES the manifest and compares the sha256 values inside it; it
never hashes the manifest file itself, which could never match.

⚠️ The record field `file_size` is None on every row these publishes produce, so it cannot be used
as an integrity check. `file_hash` is the only usable field on the record itself.

Usage:
    python tools/audit/verify_shelf_copy.py <run_id> [--ensemble chrome_box] [--sample 6]

Exit 0 verified, 1 not verified, 2 cannot run (missing coordinates — refuses rather than passing).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REGISTRY = (REPO.parent / "views-appwrite/docs/ADRs/platform/coordinate_registry.toml")
SHARD_RE = re.compile(r"__m(\d{6})\.tap\.zip$")
REQUIRED = (
    # ⛔ Taken verbatim from tools/shelf/publish_chrome_box_20261112.py. The first version of this
    # list was GUESSED from the APPWRITE_DATASTORE_* prefix and 7 of its 9 names were wrong; the
    # refusal gate below caught it rather than letting a 2-of-9 run report a pass.
    "APPWRITE_ENDPOINT",
    "APPWRITE_DATASTORE_PROJECT_ID",
    "APPWRITE_DATASTORE_API_KEY",
    "APPWRITE_PROD_FORECASTS_BUCKET_ID",
    "APPWRITE_PROD_FORECASTS_BUCKET_NAME",
    "APPWRITE_PROD_FORECASTS_COLLECTION_ID",
    "APPWRITE_PROD_FORECASTS_COLLECTION_NAME",
    "APPWRITE_METADATA_DATABASE_ID",
    "APPWRITE_METADATA_DATABASE_NAME",
)


# ⚠️ WET, with a named trigger. This loader is the SECOND copy; the first is in
# tools/shelf/publish_chrome_box_20261112.py. They are kept apart deliberately: that one is a dated
# one-shot that is meant to be deleted, and a durable audit tool must not import from a file whose
# whole purpose is to disappear. TRIGGER TO EXTRACT: a THIRD copy appearing anywhere, at which point
# it belongs in tools/credentials/ beside registry_to_env.py.
def load_coordinates() -> int:
    out = subprocess.run(
        [sys.executable, "tools/credentials/registry_to_env.py", str(REGISTRY)],
        capture_output=True, text=True, check=True, cwd=REPO,
    ).stdout
    n = 0
    for line in out.splitlines():
        if re.match(r"^[A-Z][A-Z0-9_]*=", line):  # first '=' only; values contain spaces
            key, value = line.split("=", 1)
            os.environ[key] = value.strip().strip('"').strip("'")
            n += 1
    return n


def load_secret() -> None:
    env = REPO / ".env"
    if not env.is_file():
        return
    for line in env.read_text().splitlines():
        if line.strip().startswith("APPWRITE_DATASTORE_API_KEY="):
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


def fetch(ds, file_id: str) -> bytes | None:
    """Bytes straight from the server, or None. Never from the local cache — see the trap above."""
    res = ds.download_prediction(file_id, use_cache=False, validate_cache=False)
    if not res.success or not isinstance(res.data, dict):
        return None
    if res.data.get("from_cache"):
        return None  # a cached answer cannot verify the server, so it is not an answer
    blob = res.data.get("file_bytes")
    return bytes(blob) if isinstance(blob, (bytes, bytearray)) else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_id")
    ap.add_argument("--ensemble", default="chrome_box")
    ap.add_argument("--sample", type=int, default=6,
                    help="shards to download and re-hash (0 runs check (a) only and CANNOT verify)")
    args = ap.parse_args(argv)

    os.chdir(REPO)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    if not REGISTRY.is_file():
        print(f"REFUSING: no coordinate registry at {REGISTRY}", file=sys.stderr)
        return 2
    print(f"coordinates loaded: {load_coordinates()}")
    load_secret()
    absent = [v for v in REQUIRED if not os.environ.get(v)]
    print(f"required present:   {len(REQUIRED) - len(absent)}/{len(REQUIRED)}")
    if absent:
        print(f"REFUSING rather than reporting a partial pass: missing {absent}", file=sys.stderr)
        return 2

    from views_pipeline_core.configs.prediction_store import PredictionStoreConfig
    from views_pipeline_core.managers.ensemble import EnsemblePathManager
    from views_pipeline_core.modules.datastore import DatastoreModule

    cfg = PredictionStoreConfig.from_environment()
    ds = DatastoreModule(appwrite_file_manager_config=cfg.to_appwrite_config(
        EnsemblePathManager(ensemble_name_or_path=args.ensemble, validate=False)))

    rows = [r for r in ds.list_all_predictions_unfiltered()
            if args.run_id in str(r.get("name", ""))]
    shards = {str(r["name"]): r for r in rows if SHARD_RE.search(str(r.get("name", "")))}
    man_rows = {str(r["name"]): r for r in rows
                if str(r.get("name", "")).endswith("__manifest.json")}
    fails: list[str] = []

    print("\nSHAPE")
    print(f"  files carrying the run id: {len(rows)}")
    print(f"  shard files:    {len(shards)}")
    print(f"  manifest files: {len(man_rows)}")
    if not man_rows:
        print("REFUSING: no manifest on the shelf for this run id. Under ADR-013 §3.2 the manifest "
              "is the commit marker, so this run is not published, not merely unverified.",
              file=sys.stderr)
        return 1

    # The manifests come FROM THE SERVER, not from a local log: the thing being verified must not
    # also be the source of the expectation.
    declared: dict[str, str] = {}
    for name, row in sorted(man_rows.items()):
        blob = fetch(ds, row["fileId"])
        if blob is None:
            fails.append(f"{name}: could not be fetched from the server")
            print(f"  BAD {name}: could not fetch")
            continue
        man = json.loads(blob.decode())
        months = set(man.get("expected_months", []))
        got = {int(SHARD_RE.search(k).group(1)) for k in shards
               if f"__{man.get('target')}__" in k}
        gap = sorted(months - got)
        extra = sorted(got - months)
        for s in man.get("shards", []):
            declared[s["name"]] = s["sha256"]
        ok = got == months and len(man.get("shards", [])) == len(months)
        print(f"  {'ok ' if ok else 'BAD'} {man.get('target')}: {len(got)} shards on the shelf, "
              f"manifest expects {len(months)} months "
              f"{min(months) if months else '-'}..{max(months) if months else '-'}"
              + (f", MISSING {gap}" if gap else "") + (f", UNEXPECTED {extra}" if extra else ""))
        if not ok:
            fails.append(f"{man.get('target')}: shelf does not match its manifest")

    print("\n(a) SERVER-RECORDED HASH vs MANIFEST, every shard")
    compared = agree = nohash = 0
    for name, row in sorted(shards.items()):
        want = declared.get(name)
        if not want:
            fails.append(f"{name}: on the shelf but in no manifest")
            continue
        compared += 1
        got = str(row.get("file_hash") or "")
        if not got:
            nohash += 1
        elif got == want:
            agree += 1
        else:
            fails.append(f"{name}: server hash != manifest sha256")
    print(f"  compared: {compared}; agree: {agree}; server stored no hash: {nohash}")
    unmanifested = [n for n in shards if n not in declared]
    if unmanifested:
        print(f"  BAD {len(unmanifested)} shard(s) on the shelf appear in no manifest: "
              f"{unmanifested[:3]}")
    if nohash:
        print(f"  ⚠️ {nohash} shard(s) carry no server-side hash, so (a) cannot speak for them")
        fails.append(f"{nohash} shards have no server-side hash")

    print("\n(b) REAL BYTES, fetched with the cache DISABLED, re-hashed here")
    sample = sorted(shards)[:: max(1, len(shards) // args.sample)][:args.sample] if args.sample else []
    bytes_ok = 0
    for name in sample:
        blob = fetch(ds, shards[name]["fileId"])
        if blob is None:
            print(f"  BAD {name}: no uncached bytes returned")
            fails.append(f"{name}: could not fetch uncached bytes")
            continue
        got = hashlib.sha256(blob).hexdigest()
        if got == declared.get(name):
            bytes_ok += 1
            print(f"  ok  {name}  {len(blob):,} bytes  sha256 matches the manifest")
        else:
            print(f"  BAD {name}  {len(blob):,} bytes  sha256 DIFFERS FROM THE MANIFEST")
            fails.append(f"{name}: bytes do not verify")

    print()
    if fails or not sample or bytes_ok != len(sample):
        print("NOT VERIFIED:")
        for f in fails[:15]:
            print(f"  - {f}")
        if not sample:
            print("  - no shard was downloaded, so nothing here verifies the actual bytes")
        return 1
    print(f"VERIFIED. {len(shards)} shards and {len(man_rows)} manifests under {args.run_id}. "
          f"Every manifest's expected months are present. {agree} of {compared} shards' "
          f"server-recorded hash equals the manifest sha256. {bytes_ok} shard(s) were fetched "
          f"uncached and their real bytes hash to what the manifest declares.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
