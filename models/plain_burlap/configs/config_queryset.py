"""Data specification for plain_burlap (datafactory consumer), at grid-month.

A smoothed climatology baseline, views-models#695. Fetches from the VIEWS data factory via
load_dataset() rather than connecting to PRIO's PostgreSQL through viewser.

Prerequisites:
    pip install "views-datafactory[pandas]"
    ~/.netrc entry for 204.168.219.108 (see README.md for setup)
"""

from __future__ import annotations

from datafactory_query.defaults import DEFAULT_REMOTE
from views_pipeline_core.managers.model import ModelPathManager

model_name = ModelPathManager.get_model_name_from_path(__file__)

# Data source URL — load_dataset() detects zarr vs npy from the path.
# Zarr over HTTP requires ~/.netrc credentials (see README.md).
ZARR_URL = DEFAULT_REMOTE.zarr_url

# ⛔ `land` (64,818 cells), NOT `land_gaul` (64,742) — and this is a DELIBERATE divergence
# from `ashen_sparrow`, which this file was copied from.
#
# `ashen_sparrow` uses `land_gaul` because epic #595 reconciles its pgm ensemble to a cm twin,
# and a cell GAUL gives no country to cannot be summed into one. ⇒ That reason does not apply
# here: none of #695's ten arms joins an ensemble and none is reconciled. views-baseline#109
# compares these against views-hydranet runs, which are on `land`, and a comparison across two
# different cell populations is not a comparison.
#
# The 76-cell difference is small islands outside FAO GAUL 2024 polygon coverage
# (`gaul0_code` of -1). Measured 2026-10-22 on views-datafactory 1.13.0.
REGION = "land"

# Factory name → VIEWSER name (so downstream model code doesn't change)
FEATURE_RENAME = {
    "ged_sb_best": "lr_sb_best",   # state-based fatalities (best estimate)
    "ged_ns_best": "lr_ns_best",   # non-state fatalities
    "ged_os_best": "lr_os_best",   # one-sided violence fatalities
    "gaul0_code": "c_id",          # FAO GAUL country code → identity column
}


def generate():
    """Data source descriptor (satisfies ModelPathManager.get_queryset() interface)."""
    return {
        "name": model_name,
        "source": "views-datafactory",  # "views-datafactory" or "viewser"
        "zarr_url": ZARR_URL,
        "region": REGION,               # any datafactory_query region name
        "loa": "priogrid_month",  # "priogrid_month" or "country_month"
        "features": FEATURE_RENAME,
    }
