#!/usr/bin/env python3
"""One-shot: publish the 2026-11-02 chrome_box pool to the production shelf.

A thin, dated wrapper around `tools/shelf/publish_pooled_forecast.py`. It exists because the
publish needs nine Appwrite coordinates that live in two places — eight non-secret ones in the
owned registry (`views-appwrite/.../coordinate_registry.toml`, read via
`tools/credentials/registry_to_env.py`, þing-01's read-don't-copy rule) and one secret in
`.env` — and loading them is the only fiddly part.

⛔ IT LOADS THEM IN-PROCESS, NOT THROUGH THE SHELL. A first attempt sourced the reader's
`NAME=value` output in bash and got **6 of 9**: registry values legitimately contain spaces
("Production Forecasts"), so the shell split them. The publish tool refused rather than
building a half-configured store, which is the behaviour that is supposed to happen — but the
fix is to split on the FIRST `=` only and never involve a shell.

⚠️ RUN IT WITH THE 3.4.0 VENV, NOT THE DEFAULT PYTHON. The shard provenance records
`pipeline_core_version`, defaulting to self-report. `envs/views_pipeline` reports **2.3.0**
(a stale editable install, metadata from 2026-08) over checkout code that declares 3.3.5 —
so publishing from there would stamp 2.3.0 into 108 shards produced by a pod running 3.4.0.
The venv path below is pinned for that reason.

WHAT IT PUBLISHES, and why each value is what it is:

    pool      ensembles/chrome_box/data/generated/pod_delivery_20261102/
                parquet/predictions_forecasting_20261102_112346
    run_id    chrome_box_forecasting_20261102_112346   (derived from that directory name —
              the shape `prediction_frame_ensemble.py` produces, which views-postprocessing's
              `source_selection` matches on)
    targets   lr_sb_best, lr_ns_best, lr_os_best       (renamed to lr_ged_* downstream)
    draws     1,024 = 8 members x 128, 2,330,712 rows each
    reconciled=True   — and this is the claim that matters. The run measured
              **0.000% unplaced on all three targets, scaling_failures 0**, verified against
              10,811,914 mapping rows and 202 countries. The underlying tool REFUSES to run
              unless `RECONCILIATION_RESIDUAL` carries the `✅ FULLY RECONCILED` verdict, so
              this flag cannot be asserted over a short pool (C-192, C-145).

This does NOT deliver to CRAF'd. `deliveries/un_crafd.py` still declares
`intent = paused(...)`; this only puts the forecast on the shelf the postprocessors read.
"""

from __future__ import annotations

import os
import pathlib
import re
import runpy
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
REGISTRY = (
    REPO.parent / "views-appwrite" / "docs" / "ADRs" / "platform" / "coordinate_registry.toml"
)
POOL = (
    "ensembles/chrome_box/data/generated/pod_delivery_20261102/"
    "parquet/predictions_forecasting_20261102_112346"
)

#: The nine `PredictionStoreConfig.from_environment()` requires. Listed so a missing one is
#: named here, in seconds, rather than surfacing as a ConfigurationException mid-upload.
REQUIRED = (
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


def load_coordinates() -> int:
    """Non-secret coordinates, from the owned registry via its own canonical reader."""
    out = subprocess.run(
        [sys.executable, "tools/credentials/registry_to_env.py", str(REGISTRY)],
        capture_output=True, text=True, check=True, cwd=REPO,
    ).stdout
    n = 0
    for line in out.splitlines():
        # ⛔ First '=' only. Values contain spaces; a shell `source` of this gets 6 of 9.
        if re.match(r"^[A-Z][A-Z0-9_]*=", line):
            key, value = line.split("=", 1)
            os.environ[key] = value.strip().strip('"').strip("'")
            n += 1
    return n


def load_secret() -> None:
    """The one secret, from `.env`. Never printed, never copied anywhere."""
    env = REPO / ".env"
    if not env.is_file():
        return
    for line in env.read_text().splitlines():
        if line.strip().startswith("APPWRITE_DATASTORE_API_KEY="):
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


def main() -> int:
    os.chdir(REPO)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    print(f"registry: {REGISTRY}")
    if not REGISTRY.is_file():
        print(f"!!! registry not found at {REGISTRY}", file=sys.stderr)
        return 1
    print(f"coordinates loaded: {load_coordinates()}")
    load_secret()

    missing = [v for v in REQUIRED if not os.environ.get(v)]
    print(f"required present:   {len(REQUIRED) - len(missing)}/{len(REQUIRED)}")
    if missing:
        print(f"!!! missing: {missing}", file=sys.stderr)
        return 1

    import importlib.metadata as md

    version = md.version("views-pipeline-core")
    print(f"pipeline-core:      {version}  (stamped into every shard's provenance)")
    if not version.startswith("3.4."):
        # ⛔ Refuse rather than mislabel. See the module docstring: the default python here
        # reports 2.3.0, and the pod produced this pool with 3.4.0.
        print(
            f"!!! REFUSING: this interpreter reports views-pipeline-core {version}, but the "
            f"pool was produced by 3.4.0, and the publisher stamps its own self-reported "
            f"version into all 108 shards. Run this with the 3.4.0 venv.",
            file=sys.stderr,
        )
        return 1
    print()

    # Resume support: `RESUME_TARGETS=lr_ns_best,lr_os_best bash /tmp/pubshelf.sh`.
    # ⛔ The 2026-11-02 first attempt committed lr_sb_best and then died on an Appwrite read
    # timeout mid-way through lr_ns_best. The manifest is the commit marker and is written
    # last, so lr_ns_best was left INVISIBLE rather than partial (ADR-013 §3.2) and is safe
    # to re-publish; lr_sb_best is committed and re-uploading its 36 shards buys nothing.
    argv = ["publish_pooled_forecast", "--pool", POOL, "--publish"]
    resume = os.environ.get("RESUME_TARGETS", "").strip()
    if resume:
        print(f"RESUME_TARGETS={resume}")
        argv += ["--targets", resume]
    sys.argv = argv
    runpy.run_module("tools.shelf.publish_pooled_forecast", run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
