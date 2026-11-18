"""Data specification for rocky_valley (datafactory consumer).

Instead of a viewser Queryset, rocky_valley fetches priogrid-month data from the
VIEWS data factory via load_dataset(), delivered as a views-frames FeatureFrame
and cached as a frame directory under data/raw/.

Prerequisites:
    pip install views-datafactory
    ~/.netrc entry for the data factory server (see README.md for setup)
"""

from __future__ import annotations

from datafactory_query.defaults import DEFAULT_REMOTE
from views_pipeline_core.managers.model import ModelPathManager

model_name = ModelPathManager.get_model_name_from_path(__file__)

# Data source URL — load_dataset() detects zarr vs npy from the path.
# Zarr over HTTP requires ~/.netrc credentials.
ZARR_URL = DEFAULT_REMOTE.zarr_url

# PRIO-GRID land
REGION = "africa_me_gaul"

# Factory name → VIEWSER name. The keys are also the feature request: the whole
# grid is loaded for every requested feature before the region is cut out, so
# this list is what drives peak memory during the fetch.
FEATURE_RENAME = {
    "ged_sb_count": "ged_sb_count",
    "ged_sb_best": "lr_ged_sb",  # regression target
    "ged_ns_count": "ged_ns_count",
    "ged_ns_best": "lr_ged_ns",  # regression target
    "ged_os_count": "ged_os_count",
    "ged_os_best": "lr_ged_os",  # regression target
    "ghspop_pop_count": "ghspop_pop_count",
    "ghsbuilts_built_area": "ghsbuilts_built_area",
    "vdem_v2xcl_dmove": "vdem_v2xcl_dmove",
    "vdem_v2xeg_eqdr": "vdem_v2xeg_eqdr",
    "vdem_v2xpe_exlsocgr": "vdem_v2xpe_exlsocgr",
    "vdem_v2x_clphy": "vdem_v2x_clphy",
    "vdem_v2xcl_prpty": "vdem_v2xcl_prpty",
    "vdem_v2x_ex_military": "vdem_v2x_ex_military",
    "vdem_v2x_ex_party": "vdem_v2x_ex_party",
    "vdem_v2x_horacc": "vdem_v2x_horacc",
    "vdem_v2xnp_client": "vdem_v2xnp_client",
    "vdem_v2xnp_regcorr": "vdem_v2xnp_regcorr",
    "vdem_v2xpe_exlgeo": "vdem_v2xpe_exlgeo",
    "vdem_v2x_veracc": "vdem_v2x_veracc",
    "vdem_v2xpe_exlpol": "vdem_v2xpe_exlpol",
    "vdem_v2x_diagacc": "vdem_v2x_diagacc",
    "vdem_v2x_divparctrl": "vdem_v2x_divparctrl",
    "vdem_v2xeg_eqprotec": "vdem_v2xeg_eqprotec",
    "vdem_v2x_genpp": "vdem_v2x_genpp",
    "vdem_v2xpe_exlgender": "vdem_v2xpe_exlgender",
    "vdem_v2x_hosabort": "vdem_v2x_hosabort",
    "vdem_v2x_libdem": "vdem_v2x_libdem",
    "vdem_v2xcl_rol": "vdem_v2xcl_rol",
    "vdem_v2x_accountability": "vdem_v2x_accountability",
    "shdi_shdi": "shdi_shdi",
    "shdi_healthindex": "shdi_healthindex",
    "shdi_edindex": "shdi_edindex",
    "shdi_incindex": "shdi_incindex",
    "agri_gc": "agri_gc",
    "aquaveg_gc": "aquaveg_gc",
    "barren_gc": "barren_gc",
    "cmr_max": "cmr_max",
    "cmr_mean": "cmr_mean",
    "cmr_min": "cmr_min",
    "cmr_sd": "cmr_sd",
    "diamprim_s": "diamprim_s",
    "diamsec_s": "diamsec_s",
    "forest_gc": "forest_gc",
    "gem_s": "gem_s",
    "goldplacer_s": "goldplacer_s",
    "goldsurface_s": "goldsurface_s",
    "goldvein_s": "goldvein_s",
    "growend": "growend",
    "growstart": "growstart",
    "harvarea": "harvarea",
    "herb_gc": "herb_gc",
    "imr_max": "imr_max",
    "imr_mean": "imr_mean",
    "imr_min": "imr_min",
    "imr_sd": "imr_sd",
    "landarea": "landarea",
    "mountains_mean": "mountains_mean",
    "petroleum_s": "petroleum_s",
    "rainseas": "rainseas",
    "shrub_gc": "shrub_gc",
    "ttime_max": "ttime_max",
    "ttime_mean": "ttime_mean",
    "ttime_min": "ttime_min",
    "ttime_sd": "ttime_sd",
    "urban_gc": "urban_gc",
    "water_gc": "water_gc",
}


def generate():
    """Data source descriptor (satisfies ModelPathManager.get_queryset() interface)."""
    return {
        "name": model_name,
        "source": "views-datafactory",
        "data_format": "feature_frame",  # input arrives as a views-frames FeatureFrame
        "zarr_url": ZARR_URL,
        "region": REGION,
        "loa": "priogrid_month",  # required key; the FeatureFrame path ignores it
        "features": FEATURE_RENAME,
    }
