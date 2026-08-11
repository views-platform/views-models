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
        "    pip install 'views-datafactory>=1.9.0,<2.0.0'\n"
        "and add a ~/.netrc entry for host 204.168.219.108 (the Zarr store; see "
        "the bright_starship model README)."
    ) from e

from views_pipeline_core.managers.model import ModelPathManager

model_name = ModelPathManager.get_model_name_from_path(__file__)

ZARR_URL = DEFAULT_REMOTE.zarr_url

# Global-land: the historical actuals must cover the SAME cells the forecast does.
# rusty_bucket forecasts global land, curated at the delivery boundary to `land_gaul`
# (64,742 = land ∩ FAO-GAUL coverage). Fetching actuals at this exact region makes the
# historical frame match the forecast cell-for-cell and satisfy the delivery coverage
# gate (was `africa_me_legacy`, ~13,110 Africa+ME cells — the legacy scope).
REGION = "land_gaul"

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
        # Fetch the historical actuals as a pandas-free views_frames.FeatureFrame instead
        # of a pandas DataFrame. The migrated un_fao manager reads this via
        # pipeline-core's declared_data_format() and calls get_feature_frame() rather than
        # get_data(). This is the root fix for the global-land (land_gaul, 64,742-cell,
        # 28.4M-row) actuals OOM. NOTE: no-op until the views-postprocessing manager
        # migration lands (it must — see the un_fao frame-native issue); land together.
        "data_format": "feature_frame",
    }
