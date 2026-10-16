"""Bring a partner metadata collection up to the schema a delivery actually writes.

WHY THIS EXISTS
---------------
The Appwrite Seam Contract §5.5 orders the platform's provisioning work:

    fix C-231/C-227 -> probe a scoped key -> issue tier keys -> DECLARE SCHEMA
    -> relocate create_* -> narrow scopes

`relocate create_*` shipped (views-pipeline-core #331, 3.0.0, 2026-07-31): write-time
attribute inference was deleted from five call sites. **`declare schema` did not ship, and
it was ordered first.** So the platform removed the mechanism that built partner schemas
before declaring what a partner schema is.

`unfao` works only because its attributes were accreted by that deleted inference code.
It is grandfathered. A collection created today by the sanctioned CLI gets the seven
`FIXED_METADATA_ATTRIBUTES` and nothing else — `provisioning.py` hardcodes
`ensure_collection(metadata={})` — while the delivery writes eleven keys. The first
`create_document` then fails with `Unknown attribute: loa`.

Worse, it fails *late*: `_require_containers` checks that the collection EXISTS, never
that its schema fits, so the refusal lands after `create_file` has already put bytes in
the bucket. **Every missing attribute costs one orphaned file per shard.**

This script is the stopgap until pipeline-core declares the schema. It reaches the
payload-inferred attributes the CLI cannot, by passing a representative payload — the same
input the deleted inference code used to see on every upload, supplied once, deliberately.

Importing `AppwriteProvisioner` here is sanctioned: `tests/test_import_purity.py` (#332)
forbids the DELIVERY path from importing it, not a human-run setup script.

WHAT IT DOES NOT DO
-------------------
It does not touch permissions. On an existing collection `ensure_collection` takes the
EXISTS branch and calls `ensure_attributes` only. That matters: `crafd` currently has
`$permissions: []` (correct — API-key access only), whereas `unfao` and
`production_forecasts` carry `Role.any()` read/create/update/delete and are readable and
DELETABLE by any anonymous caller who knows the project id. Do not "make crafd match".
See pipeline-core C-292.

USAGE
-----
    python tools/credentials/provision_partner_collection.py un_crafd            # dry run
    python tools/credentials/provision_partner_collection.py un_crafd --apply

Requires the platform environment (`. tools/credentials/platform_env.sh && platform_env_load`).
"""

from __future__ import annotations

import argparse
import os
import sys

#: The payload a delivery actually writes, minus the five keys pipeline-core already
#: declares. Values are representative only — `infer_attribute_type` reads their TYPE, not
#: their content: str -> string(255), list -> the same with array=True.
#:
#: Derived by enumeration from the installed code, not by trial:
#:   sink.py:161      -> loa, name, category      (the `common` dict)
#:   crafd.py:48-52   -> type, targets, description
#: The remaining five (fileId, filename, bucketId, uploaded_at, file_hash) are set by
#: pipeline-core at file.py:2152-2158 and ARE in FIXED_METADATA_ATTRIBUTES already.
#:
#: `description` is written only by the historical leg (crafd.py:394), which uploads AFTER
#: the manifest — i.e. outside the commit marker. Omitting it would give a run where every
#: shard and the manifest succeed and only the historical artifact fails. That is the
#: run-0 failure mode; it is in here deliberately.
DELIVERY_PAYLOAD_SHAPE = {
    "loa": "pgm",
    "name": "provisioning-probe",
    "type": "sampled_forecast_shard",
    "targets": ["lr_ged_sb"],          # list -> array=True, matching unfao's live shape
    "category": "forecast",
    "description": "provisioning probe",
}

#: Consumer -> the env vars holding its coordinates. Both must be exported by
#: platform_env_load; the provisioner refuses a half-pair with COORDINATE_MISMATCH.
CONSUMERS = {
    "un_crafd": ("APPWRITE_CRAFD_COLLECTION_ID", "APPWRITE_CRAFD_COLLECTION_NAME"),
    "un_fao": ("APPWRITE_UNFAO_COLLECTION_ID", "APPWRITE_UNFAO_COLLECTION_NAME"),
}


def _fail(msg: str) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("consumer", choices=sorted(CONSUMERS))
    ap.add_argument("--apply", action="store_true",
                    help="Actually create the attributes. Without it, report and exit.")
    args = ap.parse_args()

    id_var, name_var = CONSUMERS[args.consumer]
    coll_id, coll_name = os.getenv(id_var), os.getenv(name_var)
    if not coll_id or not coll_name:
        _fail(f"{id_var} / {name_var} not exported.\n"
              f"  Run: . tools/credentials/platform_env.sh && platform_env_load")

    from views_pipeline_core.modules.appwrite.provisioning import (
        FIXED_METADATA_ATTRIBUTES,
        build_provisioner,
    )

    provisioner = build_provisioner(
        collection_override=coll_id, collection_name_override=coll_name
    )
    db_id = provisioner.config.database_id

    existing = {
        a["key"]: a
        for a in provisioner.databases.list_attributes(db_id, coll_id).get("attributes", [])
    }
    declared = {a["key"] for a in FIXED_METADATA_ATTRIBUTES}
    wanted = declared | set(DELIVERY_PAYLOAD_SHAPE)
    missing = sorted(wanted - set(existing))
    # An attribute can exist but be unusable — `processing` or `failed`. Present-but-broken
    # reads as present to a naive check and then fails the write.
    unavailable = sorted(k for k, a in existing.items() if a.get("status") != "available")

    print(f"collection : {coll_id!r} ({coll_name!r}) in database {db_id!r}")
    print(f"present    : {len(existing)}  {sorted(existing) or '[]'}")
    if unavailable:
        print(f"NOT USABLE : {unavailable}  <- status != available")
    print(f"missing    : {len(missing)}  {missing or '[]'}")
    print(f"  of which declared by pipeline-core : {sorted(set(missing) & declared) or '[]'}")
    print(f"  of which inferred from the payload : {sorted(set(missing) - declared) or '[]'}")

    if not missing and not unavailable:
        print("\nNothing to do — the schema already covers what a delivery writes.")
        return 0

    if not args.apply:
        print("\nDRY RUN. Re-run with --apply to create the missing attributes.")
        return 0

    print("\nApplying...")
    result = provisioner.ensure_collection(
        metadata=DELIVERY_PAYLOAD_SHAPE,
        collection_id=coll_id,
        collection_name=coll_name,
    )
    print(f"  success={result.success} code={result.code} error={result.error}")
    if not result.success:
        _fail("ensure_collection refused. Nothing further should be attempted until this "
              "is understood — a partial schema orphans one file per shard.")

    after = {
        a["key"]: a
        for a in provisioner.databases.list_attributes(db_id, coll_id).get("attributes", [])
    }
    still_missing = sorted(wanted - set(after))
    not_ready = sorted(k for k, a in after.items() if a.get("status") != "available")
    print(f"\nafter      : {len(after)}  {sorted(after)}")
    if still_missing:
        _fail(f"still missing after apply: {still_missing}")
    if not_ready:
        # Appwrite creates attributes asynchronously; `processing` is normal for a few
        # seconds. This is a report, not a failure — but do not deliver until it clears.
        print(f"NOT YET AVAILABLE: {not_ready} — re-run the dry check until this is empty.")
        return 2
    print("\nAll attributes present and available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
