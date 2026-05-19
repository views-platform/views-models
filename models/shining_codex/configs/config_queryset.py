"""Data specification for shining_codex (datafactory consumer — country-month).

This replaces the viewser Queryset pattern used in novel_heuristics.
Instead of connecting to PRIO's PostgreSQL via viewser, shining_codex
fetches from the VIEWS data factory via load_dataset().

Prerequisites:
    pip install views-datafactory
    ~/.netrc entry for 204.168.219.108 (see datafactory docs)
"""

from __future__ import annotations

from datafactory_query.defaults import DEFAULT_REMOTE
from views_pipeline_core.managers.model import ModelPathManager

model_name = ModelPathManager.get_model_name_from_path(__file__)

ZARR_URL = DEFAULT_REMOTE.zarr_url

REGION = "global"

FACTORY_FEATURES = ["ged_sb_best", "ged_ns_best", "ged_os_best"]

FEATURE_RENAME = {
    "ged_sb_best": "lr_ged_sb",
    "ged_ns_best": "lr_ged_ns",
    "ged_os_best": "lr_ged_os",
}


def generate():
    """Data source descriptor (satisfies ModelPathManager.get_queryset() interface)."""
    return {
        "name": model_name,
        "source": "views-datafactory",
        "zarr_url": ZARR_URL,
        "region": REGION,
        "loa": "country_month",
        "features": FEATURE_RENAME,
    }
