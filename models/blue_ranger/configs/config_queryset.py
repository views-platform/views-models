from __future__ import annotations

from datafactory_query.defaults import DEFAULT_REMOTE
from views_pipeline_core.managers.model import ModelPathManager

model_name = ModelPathManager.get_model_name_from_path(__file__)

# Data source URL — load_dataset() detects zarr vs npy from the path.
# Zarr over HTTP requires ~/.netrc credentials (see README.md).
ZARR_URL = DEFAULT_REMOTE.zarr_url

# 64,818 PRIO-GRID land cells (global coverage, excluding water)
REGION = "land"

# Factory name → VIEWSER name (so downstream model code doesn't change)
FEATURE_RENAME = {
    "ged_sb_best": "lr_ged_sb",   # state-based fatalities (best estimate)
    "ged_ns_best": "lr_ged_ns",   # non-state fatalities
    "ged_os_best": "lr_ged_os",   # one-sided violence fatalities
    # "gaul0_code": "c_id",          # FAO GAUL country code → identity column
}

def generate():
    """Data source descriptor (satisfies ModelPathManager.get_queryset() interface)."""
    return {
        "name": model_name,
        "source": "views-datafactory",  # "views-datafactory" or "viewser"
        "zarr_url": ZARR_URL,
        "region": REGION,               # any datafactory_query region name
        "loa": "priogrid_month",        # "priogrid_month" or "country_month"
        "features": FEATURE_RENAME,
    }
