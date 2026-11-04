"""Data specification for the un_crafd postprocessor (datafactory consumer).

Fetches historical UCDP fatality targets from the VIEWS data factory via the
pandas-free frame path. The forecast leg comes from the shared `production_forecasts`
shelf; this file describes only the historical actuals.

Prerequisites:
    pip install 'views-datafactory>=1.9.0,<2.0.0'
    ~/.netrc entry for 204.168.219.108 (see bright_starship README)
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from datafactory_query.defaults import DEFAULT_REMOTE
except ImportError as e:  # fail loud with the fix, not a bare ModuleNotFoundError (#95)
    raise RuntimeError(
        "The un_crafd postprocessor requires views-datafactory (provides the "
        "`datafactory_query` module), which is not installed in this environment.\n"
        "Install it:\n"
        "    pip install 'views-datafactory>=1.9.0,<2.0.0'\n"
        "and add a ~/.netrc entry for host 204.168.219.108 (the Zarr store; see "
        "the bright_starship model README)."
    ) from e

from views_pipeline_core.managers.model import ModelPathManager

model_name = ModelPathManager.get_model_name_from_path(__file__)

ZARR_URL = DEFAULT_REMOTE.zarr_url

# The delivery declaration lives at the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deliveries.status import declared_coverage  # noqa: E402  (after the path bootstrap)

# Derived, never typed (ADR-021). The actuals fetch region and the delivered coverage are
# ONE fact: the historical frame must cover the same cells as the forecast, or the
# delivery's coverage gate refuses it. Typing it here as well would give the repository
# two copies to disagree about — which is what happened to un_fao for seven weeks (C-110).
#
# To change it, edit `coverage` in deliveries/un_crafd.py; nothing here needs touching.
REGION = declared_coverage("un_crafd")

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
        # The frame path, not pandas. `views_postprocessing.crafd` refuses anything else,
        # and at global-land scale (64,742 cells x ~438 months = 28.4M rows) the pandas
        # path OOM-kills at ~24 GB.
        #
        # If you ever see the manager complain that this says `dataframe` while the line
        # above plainly says `feature_frame`, the file failed to IMPORT: pipeline-core's
        # get_queryset() swallows the exception (model_path.py:783-785), returns None, and
        # declared_data_format(None) defaults to `dataframe`. Check the import first
        # (views-postprocessing C-83).
        "data_format": "feature_frame",
    }
