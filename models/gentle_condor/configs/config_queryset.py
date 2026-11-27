"""Data specification for gentle_condor (datafactory consumer), at country-month.

A member of epic #595. Fetches from the VIEWS data factory via load_dataset()
rather than connecting to PRIO's PostgreSQL through viewser.

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

# 64,742 land cells that GAUL gives a country to — `land_gaul`, NOT `land` (64,818).
#
# The 76-cell difference is small islands outside FAO GAUL 2024 polygon coverage,
# whose `gaul0_code` is -1. Epic #595 reconciles the pgm ensemble to its cm twin and
# requires the summed grid draws to equal the country draws. A cell with no country
# cannot be summed into one: views-datafactory's CM aggregation drops
# `country_id <= 0`, and it only *warns* about dropped events when the caller
# declares `feature_agg_types`, so on `land` the discrepancy is silent.
# Measured 2026-10-22 on views-datafactory 1.13.0: land 64,818, land_gaul 64,742.
REGION = "land_gaul"
# views-datafactory sums grid cells per country by `gaul0_code`
# (datafactory_adapters/grid_to_country_month.py) to build this level, which is why
# the pandas extra is in requirements.txt — the aggregation needs it.

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
        "loa": "country_month",   # "priogrid_month" or "country_month"
        "features": FEATURE_RENAME,
    }
