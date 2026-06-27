"""Data specification for ravaging_mage (views-datafactory, country_month).

DRAFT (pending datafactory CM-aggregation finalization). country_month is served by
datafactory load_dataset (verified). The minimal UCDP set (ged_*_best, counts)
aggregates correctly by SUM at country level. Do NOT add intensive features
(V-Dem / most WDI = indices/rates) here until the datafactory per-feature
aggregation (weighted mean) lands — summing indices is meaningless (ADR-040).

Prerequisites: pip install views-datafactory; ~/.netrc for the zarr host.
"""
from __future__ import annotations

from datafactory_query.defaults import DEFAULT_REMOTE
from views_pipeline_core.managers.model import ModelPathManager

model_name = ModelPathManager.get_model_name_from_path(__file__)

ZARR_URL = DEFAULT_REMOTE.zarr_url
REGION = "land"  # global land; aggregated to all countries at country_month (verify on finalize)

# datafactory zarr field -> internal name. Minimal UCDP counts + country identity.
FEATURE_RENAME = {
    "ged_sb_best": "lr_ged_sb",
    "ged_ns_best": "lr_ged_ns",
    "ged_os_best": "lr_ged_os",
    "gaul0_code": "c_id",  # country identity (gaul0); excluded from aggregation
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
