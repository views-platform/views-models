"""Data specification for un_fao postprocessor (datafactory consumer).

Fetches historical UCDP fatality targets from the VIEWS data factory
via load_dataset(). Replaces the previous viewser Queryset pattern.

Prerequisites:
    pip install views-datafactory
    ~/.netrc entry for 204.168.219.108 (see bright_starship README)
"""

from __future__ import annotations

try:
    from datafactory_query.defaults import DEFAULT_REMOTE
except ImportError as e:  # fail loud with the fix, not a bare ModuleNotFoundError (#95)
    raise RuntimeError(
        "The un_fao postprocessor requires views-datafactory (provides the "
        "`datafactory_query` module), which is not installed in this environment.\n"
        "Install it:\n"
        "    pip install 'views-datafactory @ "
        "git+https://github.com/views-platform/views-datafactory.git@development'\n"
        "and add a ~/.netrc entry for host 204.168.219.108 (the Zarr store; see "
        "the bright_starship model README)."
    ) from e

from views_pipeline_core.managers.model import ModelPathManager

model_name = ModelPathManager.get_model_name_from_path(__file__)

ZARR_URL = DEFAULT_REMOTE.zarr_url

REGION = "africa_me_legacy"

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
        "loa": "priogrid_month",
        "features": FEATURE_RENAME,
    }
