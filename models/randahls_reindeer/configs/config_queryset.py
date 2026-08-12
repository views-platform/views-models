# from viewser import Queryset, Column
# from views_pipeline_core.managers.model import ModelPathManager

# model_name = ModelPathManager.get_model_name_from_path(__file__)

# def generate():
#     """
#     Contains the configuration for the input data in the form of a viewser queryset. That is the data from viewser that is used to train the model.
#     This configuration is "behavioral" so modifying it will affect the model's runtime behavior and integration into the deployment system.
#     There is no guarantee that the model will work if the input data configuration is changed here without changing the model settings and algorithm accordingly.

#     Returns:
#     - queryset_base (Queryset): A queryset containing the base data for the model training.
#     """
    
#     qs_markov = (Queryset(f'{model_name}','country_month')

#         # # target variable
#         # .with_column(Column("lr_ged_sb_dep", from_loa="country_month", from_column="ged_sb_best_sum_nokgi")
#         #         .transform.missing.fill()
#         #         )

#         # Baseline features:
#         # lag of target variable
#         .with_column(Column("lr_ged_sb", from_loa="country_month", from_column="ged_sb_best_sum_nokgi")
#                 .transform.missing.fill()
#                 )

#         # Decay functions
#         # sb
#         .with_column(Column("decay_ged_sb_5", from_loa="country_month", from_column="ged_sb_best_sum_nokgi")
#                 .transform.missing.replace_na()
#                 .transform.bool.gte(5)
#                 .transform.temporal.time_since()
#                 .transform.temporal.decay(24)
#                 .transform.missing.replace_na()
#                 )
#         # os
#         .with_column(Column("decay_ged_os_5", from_loa="country_month", from_column="ged_os_best_sum_nokgi")
#                 .transform.missing.replace_na()
#                 .transform.bool.gte(5)
#                 .transform.temporal.time_since()
#                 .transform.temporal.decay(24)
#                 .transform.missing.replace_na()
#                 )

#         # Spatial lag decay
#         .with_column(Column("splag_1_decay_ged_sb_5", from_loa="country_month",
#                         from_column="ged_sb_best_sum_nokgi")
#                 .transform.missing.replace_na()
#                 .transform.bool.gte(5)
#                 .transform.temporal.time_since()
#                 .transform.temporal.decay(24)
#                 .transform.spatial.countrylag(1, 1, 0, 0)
#                 .transform.missing.replace_na()
#                 )

#         .with_column(Column("wdi_sp_pop_totl", from_loa="country_year", from_column="wdi_sp_pop_totl")
#                 .transform.missing.fill()
#                 .transform.temporal.tlag(12)
#                 .transform.missing.fill()
#                 )

#         # More conflict history [hh20]
#         .with_column(Column("lr_ged_sb_tlag_1", from_loa="country_month",
#                         from_column="ged_sb_best_sum_nokgi")
#                 .transform.missing.fill()
#                 .transform.temporal.tlag(1)
#                 .transform.missing.fill()
#                 )

#         .with_column(Column("lr_ged_sb_tlag_2", from_loa="country_month",
#                         from_column="ged_sb_best_sum_nokgi")
#                 .transform.missing.fill()
#                 .transform.temporal.tlag(2)
#                 .transform.missing.fill()
#                 )

#         .with_column(Column("decay_acled_os_5", from_loa="country_month", from_column="acled_os_fat")
#                 .transform.missing.replace_na()
#                 .transform.bool.gte(5)
#                 .transform.temporal.time_since()
#                 .transform.temporal.decay(24)
#                 .transform.missing.replace_na()
#                 )

#         .with_column(Column("decay_ged_sb_100", from_loa="country_month",
#                         from_column="ged_sb_best_sum_nokgi")
#                 .transform.missing.replace_na()
#                 .transform.bool.gte(100)
#                 .transform.temporal.time_since()
#                 .transform.temporal.decay(24)
#                 .transform.missing.replace_na()
#                 )

#         .with_column(Column("decay_ged_sb_500", from_loa="country_month",
#                         from_column="ged_sb_best_sum_nokgi")
#                 .transform.missing.replace_na()
#                 .transform.bool.gte(500)
#                 .transform.temporal.time_since()
#                 .transform.temporal.decay(24)
#                 .transform.missing.replace_na()
#                 )

#         # Features from reign [hh20]
#         .with_column(Column("reign_tenure_months", from_loa="country_month", from_column="tenure_months")
#                 .transform.missing.fill()
#                 .transform.missing.replace_na()
#                 )

#         # From WDI [hh20]
#         .with_column(Column("wdi_ag_lnd_frst_k2", from_loa="country_year", from_column="wdi_ag_lnd_frst_k2")
#                 .transform.missing.fill()
#                 .transform.temporal.tlag(12)
#                 .transform.missing.fill()
#                 )

#         .with_column(Column("wdi_nv_agr_totl_kn", from_loa="country_year", from_column="wdi_nv_agr_totl_kn")
#                 .transform.missing.fill()
#                 .transform.temporal.tlag(12)
#                 .transform.missing.fill()
#                 )

#         .with_column(Column("wdi_sh_sta_maln_zs", from_loa="country_year", from_column="wdi_sh_sta_maln_zs")
#                 .transform.missing.fill()
#                 .transform.temporal.tlag(12)
#                 .transform.missing.fill()
#                 )

#         .with_column(Column("wdi_sl_tlf_totl_fe_zs", from_loa="country_year",
#                         from_column="wdi_sl_tlf_totl_fe_zs")
#                 .transform.missing.fill()
#                 .transform.temporal.tlag(12)
#                 .transform.missing.fill()
#                 )

#         .with_column(Column("wdi_sm_pop_refg_or", from_loa="country_year", from_column="wdi_sm_pop_refg_or")
#                 .transform.missing.fill()
#                 .transform.temporal.tlag(12)
#                 .transform.missing.fill()
#                 )

#         .with_column(Column("wdi_sp_dyn_imrt_in", from_loa="country_year", from_column="wdi_sp_dyn_imrt_in")
#                 .transform.missing.fill()
#                 .transform.temporal.tlag(12)
#                 .transform.missing.fill()
#                 )

#         .with_column(Column("wdi_sp_pop_14_fe_zs", from_loa="country_year",
#                         from_column="wdi_sp_pop_0014_fe_zs")
#                 .transform.missing.fill()
#                 .transform.temporal.tlag(12)
#                 .transform.missing.fill()
#                 )

#         .with_column(Column("wdi_sp_pop_grow", from_loa="country_year", from_column="wdi_sp_pop_grow")
#                 .transform.missing.fill()
#                 .transform.temporal.tlag(12)
#                 .transform.missing.fill()
#                 )

#         # Spatial lags [hh20]

#         .with_column(Column("splag_wdi_ag_lnd_frst_k2", from_loa="country_year",
#                         from_column="wdi_ag_lnd_frst_k2")
#                 .transform.missing.fill()
#                 .transform.temporal.tlag(12)
#                 .transform.spatial.countrylag(1, 1, 0, 0)
#                 .transform.missing.replace_na()
#                 )

#         .with_column(Column("splag_wdi_sl_tlf_totl_fe_zs", from_loa="country_year",
#                         from_column="wdi_sl_tlf_totl_fe_zs")
#                 .transform.missing.fill()
#                 .transform.temporal.tlag(12)
#                 .transform.spatial.countrylag(1, 1, 0, 0)
#                 .transform.missing.replace_na()
#                 )

#         .with_column(Column("splag_wdi_sm_pop_netm", from_loa="country_year", from_column="wdi_sm_pop_netm")
#                 .transform.missing.fill()
#                 .transform.temporal.tlag(12)
#                 .transform.spatial.countrylag(1, 1, 0, 0)
#                 .transform.missing.replace_na()
#                 )

#         # From Vdem
#         .with_column(Column("vdem_v2xcl_dmove", from_loa="country_year", from_column="vdem_v2xcl_dmove")
#                 .transform.missing.fill()
#                 .transform.temporal.tlag(12)
#                 .transform.missing.fill()
#                 )

#         .with_column(Column("vdem_v2xcl_rol", from_loa="country_year", from_column="vdem_v2xcl_rol")
#                 .transform.missing.fill()
#                 .transform.temporal.tlag(12)
#                 .transform.missing.fill()
#                 )

#         .with_column(Column("vdem_v2xeg_eqdr", from_loa="country_year", from_column="vdem_v2xeg_eqdr")
#                 .transform.missing.fill()
#                 .transform.temporal.tlag(12)
#                 .transform.missing.fill()
#                 )

#         .with_column(Column("vdem_v2xpe_exlpol", from_loa="country_year",
#                         from_column="vdem_v2xpe_exlpol")
#                 .transform.missing.fill()
#                 .transform.temporal.tlag(12)
#                 .transform.missing.fill()
#                 )

#         .with_column(Column("vdem_v2xpe_exlsocgr", from_loa="country_year",
#                         from_column="vdem_v2xpe_exlsocgr")
#                 .transform.missing.fill()
#                 .transform.temporal.tlag(12)
#                 .transform.missing.fill()
#                 )

#         .with_column(Column("splag_vdem_v2xpe_exlsocgr", from_loa="country_year",
#                         from_column="vdem_v2xpe_exlsocgr")
#                 .transform.missing.fill()
#                 .transform.temporal.tlag(12)
#                 .transform.spatial.countrylag(1, 1, 0, 0)
#                 .transform.missing.replace_na()
#                 )

#         .with_column(Column("splag_vdem_v2xcl_rol", from_loa="country_year",
#                         from_column="vdem_v2xcl_rol")
#                 .transform.missing.fill()
#                 .transform.temporal.tlag(12)
#                 .transform.spatial.countrylag(1, 1, 0, 0)
#                 .transform.missing.replace_na()
#                 )

#         .with_theme("fatalities")
#         .describe("""Markov Queryset based on "fatalities002_joint_narrow" used by the original Markov models written in R.""")
#         )

#     return qs_markov

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
    # "gaul0_code": "c_id",          # FAO GAUL country code → identity column
}

def generate():
    """Data source descriptor (satisfies ModelPathManager.get_queryset() interface)."""
    return {
        "name": model_name,
        "source": "views-datafactory",  # "views-datafactory" or "viewser"
        "zarr_url": ZARR_URL,
        "region": REGION,               # any datafactory_query region name
        "loa": "country_month",        # "priogrid_month" or "country_month"
        "features": FEATURE_RENAME,
    }
