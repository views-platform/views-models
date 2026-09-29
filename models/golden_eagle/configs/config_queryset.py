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
    "agri_gc": "lr_agri_gc",
    "acled_battles": "lr_acled_battles",
    "acled_explosions": "lr_acled_explosions",
    "acled_vac": "lr_acled_vac",
    "acled_protests": "lr_acled_protests",
    "acled_riots": "lr_acled_riots",
    "acled_strategic": "lr_acled_strategic",
    "acled_fatalities": "lr_acled_fatalities",
    "aquaveg_gc": "lr_aquaveg_gc",
    "barren_gc": "lr_barren_gc",
    "cmr_max": "lr_cmr_max",
    "cmr_mean": "lr_cmr_mean",
    "cmr_min": "lr_cmr_min",
    "cmr_sd": "lr_cmr_sd",
    "diamprim_s": "lr_diamprim_s",
    "diamsec_s": "lr_diamsec_s",
    "forest_gc": "lr_forest_gc",
    "gem_s": "lr_gem_s",
    "goldplacer_s": "lr_goldplacer_s",
    "goldsurface_s": "lr_goldsurface_s",
    "goldvein_s": "lr_goldvein_s",
    "growend": "lr_growend",
    "growstart": "lr_growstart",
    "harvarea": "lr_harvarea",
    "herb_gc": "lr_herb_gc",
    "ghspop_pop_count": "lr_ghspop_pop_count",
    "ghsbuilts_built_area": "lr_ghsbuilts_built_area",
    "imr_max": "lr_imr_max",
    "imr_mean": "lr_imr_mean",
    "imr_min": "lr_imr_min",
    "imr_sd": "lr_imr_sd",
    "landarea": "lr_landarea",
    "maincrop": "lr_maincrop",
    "mountains_mean": "lr_mountains_mean",
    "petroleum_s": "lr_petroleum_s",
    "rainseas": "lr_rainseas",
    "shdi_shdi": "lr_shdi_shdi",
    "shdi_healthindex": "lr_shdi_healthindex",
    "shdi_edindex": "lr_shdi_edindex",
    "shdi_incindex": "lr_shdi_incindex",
    "shrub_gc": "lr_shrub_gc",
    "ttime_max": "lr_ttime_max",
    "ttime_mean": "lr_ttime_mean",
    "ttime_min": "lr_ttime_min",
    "ttime_sd": "lr_ttime_sd",
    "urban_gc": "lr_urban_gc",
    "vdem_v2xcl_dmove": "lr_vdem_v2xcl_dmove",
    "vdem_v2xeg_eqdr": "lr_vdem_v2xeg_eqdr",
    "vdem_v2xpe_exlsocgr": "lr_vdem_v2xpe_exlsocgr",
    "vdem_v2x_clphy": "lr_vdem_v2x_clphy",
    "vdem_v2xcl_prpty": "lr_vdem_v2xcl_prpty",
    "vdem_v2x_ex_military": "lr_vdem_v2x_ex_military",
    "vdem_v2x_ex_party": "lr_vdem_v2x_ex_party",
    "vdem_v2x_horacc": "lr_vdem_v2x_horacc",
    "vdem_v2xnp_client": "lr_vdem_v2xnp_client",
    "vdem_v2xnp_regcorr": "lr_vdem_v2xnp_regcorr",
    "vdem_v2xpe_exlgeo": "lr_vdem_v2xpe_exlgeo",
    "vdem_v2x_veracc": "lr_vdem_v2x_veracc",
    "vdem_v2xpe_exlpol": "lr_vdem_v2xpe_exlpol",
    "vdem_v2x_diagacc": "lr_vdem_v2x_diagacc",
    "vdem_v2x_divparctrl": "lr_vdem_v2x_divparctrl",
    "vdem_v2xeg_eqprotec": "lr_vdem_v2xeg_eqprotec",
    "vdem_v2x_genpp": "lr_vdem_v2x_genpp",
    "vdem_v2xpe_exlgender": "lr_vdem_v2xpe_exlgender",
    "vdem_v2x_hosabort": "lr_vdem_v2x_hosabort",
    "vdem_v2x_libdem": "lr_vdem_v2x_libdem",
    "vdem_v2xcl_rol": "lr_vdem_v2xcl_rol",
    "vdem_v2x_accountability": "lr_vdem_v2x_accountability",
    "water_gc": "lr_water_gc",

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
