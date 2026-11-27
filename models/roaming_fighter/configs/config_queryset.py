"""Data specification for roaming_fighter (views-datafactory, country_month).

country_month is served by datafactory load_dataset (ADR-048: registry-declared
feature_agg_types; intensive-at-CM fails loud). The minimal UCDP set
(ged_*_best, counts) aggregates correctly by SUM at country level. Do NOT add
intensive features (V-Dem / most WDI = indices/rates) here: over the REMOTE
zarr the ADR-048 guard is inactive (feature_agg_types=None — register C-94),
so summed indices would be silently meaningless.

Prerequisites: pip install views-datafactory; ~/.netrc for the zarr host.
"""
from __future__ import annotations

from datafactory_query.defaults import DEFAULT_REMOTE
from views_pipeline_core.managers.model import ModelPathManager

model_name = ModelPathManager.get_model_name_from_path(__file__)

ZARR_URL = DEFAULT_REMOTE.zarr_url
REGION = "land"  # global land (64,818 cells); country_month drops the 76 GAUL-unmapped cells

# datafactory zarr field -> internal name. Minimal UCDP counts (gaul0_code is auto-added by datafactory for CM grouping and dropped from output).
FEATURE_RENAME = {
    "ged_sb_best": "lr_ged_sb",
    "ged_ns_best": "lr_ged_ns",
    "ged_os_best": "lr_ged_os",
}


def generate():
    """Data source descriptor (satisfies ModelPathManager.get_queryset())."""
    return {
        "name": model_name,
        "source": "views-datafactory",
        "zarr_url": ZARR_URL,
        "region": REGION,
        "loa": "country_month",
        "features": FEATURE_RENAME,
    }
