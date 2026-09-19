"""Data specification for un_fao postprocessor (datafactory consumer).

Fetches historical UCDP fatality targets from the VIEWS data factory
via load_dataset(). Replaces the previous viewser Queryset pattern.

Prerequisites:
    pip install views-datafactory
    ~/.netrc entry for 204.168.219.108 (see bright_starship README)
"""

from __future__ import annotations

import sys
from pathlib import Path

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

# Derived, never typed (ADR-021). The actuals fetch region and the delivered coverage
# are ONE fact: the historical frame must cover the same cells as the forecast, which
# is why this was set to the delivery's region in the first place. Typing it here as
# well gave the repository two copies to disagree about, and it did — `africa_me_legacy`
# in git against `land_gaul` in a working tree, for seven weeks (register C-110).
#
# Today: `land_gaul`, 64,742 cells = global land ∩ FAO-GAUL coverage. To change it,
# edit `coverage` in deliveries/un_fao.py; nothing here needs touching.
#
# Not to be confused with the producer's extent: rusty_bucket forecasts `land` (64,818)
# and the delivery boundary removes 76 sub-Antarctic cells outside GAUL 2024. That
# curation belongs to views_postprocessing/delivery/coverage.py.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deliveries.status import declared_coverage  # noqa: E402  (after the path bootstrap)

REGION = declared_coverage("un_fao")

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
