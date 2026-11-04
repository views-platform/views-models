"""Data specification for violet_visitor (views-datafactory consumer).

Migrated from viewser to views-datafactory per ADR-071 / Epic #203. Instead of connecting to PRIO's
PostgreSQL via viewser, violet_visitor fetches from the VIEWS data factory via load_dataset(), which
handles Known Geographical Imprecision (KGI) that viewser's legacy `_sum_nokgi` targets left
unhandled. Conflict-target parity vs viewser is validated Tier-A (dossier 07 E1; PASS on a fresh pull).

Prerequisites:
    pip install "views-datafactory>=1.9.0"
    ~/.netrc entry for 204.168.219.108 (see README.md for setup)

The previous viewser queryset is preserved in the migration dossier's evidence trail.
"""

from __future__ import annotations

from datafactory_query.defaults import DEFAULT_REMOTE

# ⛔ PINNED by make_ss_arm.py, and NOT a copy-paste defect — see C-194.
#
# The descriptor's `name` is an INPUT to `queryset_digest`. Measured 2026-11-04: violet_visitor,
# copper_kestrel, indigo_kestrel and silver_kestrel have byte-identical descriptors and four
# DIFFERENT digests, because each derives its own name from its own path; these twelve arms pin
# the name and all share digest b32a221b…, as violet_visitor does.
#
# `tools/podrun/pod_run_exp02_arm.sh:86` pins ONE EXPECTED_DIGEST and checks every arm against it
# ~430 s into a run, before any GPU spend. Deriving this name from the path would give each arm a
# different digest and the gate would refuse eleven of twelve healthy arms.
#
# ⚠️ So an audit reporting `name != directory` here is reporting the CORRECT state. Do not
# "fix" it. Guarded by tests/test_exp_arms_share_the_floor_queryset_identity.py.
model_name = "violet_visitor"

# Data source URL — load_dataset() detects zarr vs npy from the path.
# Zarr over HTTP requires ~/.netrc credentials (see README.md).
ZARR_URL = DEFAULT_REMOTE.zarr_url

# 64,818 PRIO-GRID land cells (global coverage, excluding water) — runbook #499 Step 1;
# africa_me_legacy (13,110 cells) until 2026-09-19. The FAO delivery cuts land_gaul from this.
REGION = "land"

# Factory name → VIEWSER name (so downstream model code / configs don't change).
FEATURE_RENAME = {
    "ged_sb_best": "lr_sb_best",   # state-based fatalities (best estimate)
    "ged_ns_best": "lr_ns_best",   # non-state fatalities
    "ged_os_best": "lr_os_best",   # one-sided violence fatalities
    "gaul0_code": "c_id",          # FAO GAUL country code → identity column
}


def generate():
    """Data source descriptor (satisfies ModelPathManager.get_queryset() interface).

    Returns a dict descriptor (NOT a viewser Queryset). views-pipeline-core's ViewsDataLoader
    dispatches on `source` and routes this to the datafactory fetch path (ADR-050 consumer contract).
    """
    return {
        "name": model_name,
        "source": "views-datafactory",  # "views-datafactory" or "viewser"
        "zarr_url": ZARR_URL,
        "region": REGION,               # any datafactory_query region name
        "loa": "priogrid_month",        # "priogrid_month" or "country_month"
        "features": FEATURE_RENAME,
    }
