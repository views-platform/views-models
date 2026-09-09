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
from views_pipeline_core.managers.model import ModelPathManager

model_name = ModelPathManager.get_model_name_from_path(__file__)

# Data source URL — load_dataset() detects zarr vs npy from the path.
# Zarr over HTTP requires ~/.netrc credentials (see README.md).
ZARR_URL = DEFAULT_REMOTE.zarr_url

# 13,110 PRIO-GRID cells matching VIEWSER's Africa + Middle East coverage.
REGION = "africa_me_legacy"

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
