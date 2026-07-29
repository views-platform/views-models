"""Data specification

This replaces the viewser Queryset pattern used in other models.
Instead of connecting to PRIO's PostgreSQL via viewser, heavy_freighter
fetches from the VIEWS data factory via load_dataset().

Prerequisites:
    pip install views-datafactory
    ~/.netrc entry for 204.168.219.108 (see README.md for setup)
"""

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
        "loa": "priogrid_month",        # "priogrid_month" or "country_month"
        "features": FEATURE_RENAME,
    }

# from viewser import Queryset, Column
# from views_pipeline_core.managers.model import ModelPathManager

# model_name = ModelPathManager.get_model_name_from_path(__file__)


# def generate():
    
#     qs_natsoc = (Queryset(f'{model_name}','priogrid_month')

#                 .with_column(Column('lr_ged_sb', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
#                     .transform.missing.fill()
#                     .transform.missing.replace_na()
#                     )

#                 .with_column(Column('lr_ged_os', from_loa='priogrid_month', from_column='ged_os_best_sum_nokgi')
#                     .transform.missing.fill()
#                     .transform.missing.replace_na()
#                     )

#                 .with_column(Column('lr_ged_ns', from_loa='priogrid_month', from_column='ged_ns_best_sum_nokgi')
#                     .transform.missing.fill()
#                     .transform.missing.replace_na()
#                     )

#                 # .with_column(Column('lr_treelag_1_sb', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.spatial.treelag(0.7,1)
#                 #     )

#                 # .with_column(Column('lr_treelag_2_sb', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.spatial.treelag(0.7,2)
#                 #     )

#                 # .with_column(Column('lr_treelag_1_os', from_loa='priogrid_month', from_column='ged_os_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.spatial.treelag(0.7,1)
#                 #     )

#                 # .with_column(Column('lr_treelag_2_os', from_loa='priogrid_month', from_column='ged_os_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.spatial.treelag(0.7,2)
#                 #     )

#                 # .with_column(Column('lr_treelag_1_ns', from_loa='priogrid_month', from_column='ged_ns_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.spatial.treelag(0.7,1)
#                 #     )

#                 # .with_column(Column('lr_treelag_2_ns', from_loa='priogrid_month', from_column='ged_ns_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.spatial.treelag(0.7,2)
#                 #     )

#                 # .with_column(Column('lr_sptime_dist_k1_ged_sb', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.spatial.sptime_dist('distances',1,1.0,0.0)
#                 #     )

#                 # .with_column(Column('lr_sptime_dist_k10_ged_sb', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.spatial.sptime_dist('distances',1,10.0,0.0)
#                 #     )

#                 # .with_column(Column('lr_sptime_dist_k001_ged_sb', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.spatial.sptime_dist('distances',1,0.01,0.0)
#                 #     )

#                 # .with_column(Column('lr_sptime_dist_k1_ged_os', from_loa='priogrid_month', from_column='ged_os_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.spatial.sptime_dist('distances',1,1.0,0.0)
#                 #     )

#                 # .with_column(Column('lr_sptime_dist_k10_ged_os', from_loa='priogrid_month', from_column='ged_os_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.spatial.sptime_dist('distances',1,10.0,0.0)
#                 #     )

#                 # .with_column(Column('lr_sptime_dist_k001_ged_os', from_loa='priogrid_month', from_column='ged_os_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.spatial.sptime_dist('distances',1,0.01,0.0)
#                 #     )

#                 # .with_column(Column('lr_sptime_dist_k1_ged_ns', from_loa='priogrid_month', from_column='ged_ns_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.spatial.sptime_dist('distances',1,1.0,0.0)
#                 #     )

#                 # .with_column(Column('lr_sptime_dist_k10_ged_ns', from_loa='priogrid_month', from_column='ged_ns_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.spatial.sptime_dist('distances',1,10.0,0.0)
#                 #     )

#                 # .with_column(Column('lr_sptime_dist_k001_ged_ns', from_loa='priogrid_month', from_column='ged_ns_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.spatial.sptime_dist('distances',1,0.01,0.0)
#                 #     )

#                 # Natural and Social Geography features (disabled)
#                 # .with_column(Column('lr_imr_mean', from_loa='priogrid_year', from_column='imr_mean')
#                 #     .transform.missing.fill()
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('lr_mountains_mean', from_loa='priogrid_year', from_column='mountains_mean')
#                 #     .transform.missing.fill()
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('lr_dist_diamsec', from_loa='priogrid', from_column='dist_diamsec_s_wgs')
#                 #     .transform.missing.fill()
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('lr_dist_petroleum', from_loa='priogrid', from_column='dist_petroleum_s_wgs')
#                 #     .transform.missing.fill()
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('lr_agri_ih', from_loa='priogrid_year', from_column='agri_ih')
#                 #     .transform.missing.fill()
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('lr_barren_ih', from_loa='priogrid_year', from_column='barren_ih')
#                 #     .transform.missing.fill()
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('lr_forest_ih', from_loa='priogrid_year', from_column='forest_ih')
#                 #     .transform.missing.fill()
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('lr_pasture_ih', from_loa='priogrid_year', from_column='pasture_ih')
#                 #     .transform.missing.fill()
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('lr_savanna_ih', from_loa='priogrid_year', from_column='savanna_ih')
#                 #     .transform.missing.fill()
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('lr_shrub_ih', from_loa='priogrid_year', from_column='shrub_ih')
#                 #     .transform.missing.fill()
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('lr_urban_ih', from_loa='priogrid_year', from_column='urban_ih')
#                 #     .transform.missing.fill()
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('ln_pop_gpw_sum', from_loa='priogrid_year', from_column='pop_gpw_sum')
#                 #     .transform.ops.ln()
#                 #     .transform.missing.fill()
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('ln_ttime_mean', from_loa='priogrid_year', from_column='ttime_mean')
#                 #     .transform.ops.ln()
#                 #     .transform.missing.fill()
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('ln_gcp_mer', from_loa='priogrid_year', from_column='gcp_mer')
#                 #     .transform.ops.ln()
#                 #     .transform.missing.fill()
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('ln_bdist3', from_loa='priogrid_year', from_column='bdist3')
#                 #     .transform.ops.ln()
#                 #     .transform.missing.fill()
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('ln_capdist', from_loa='priogrid_year', from_column='capdist')
#                 #     .transform.ops.ln()
#                 #     .transform.missing.fill()
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('lr_greq_1_excluded', from_loa='priogrid_year', from_column='excluded')
#                 #     .transform.bool.gte(1)
#                 #     .transform.missing.fill()
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # Conflict decay memory features (disabled)
#                 # .with_column(Column('lr_decay_ged_sb_1', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.bool.gte(1)
#                 #     .transform.temporal.time_since()
#                 #     .transform.temporal.decay(24)
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('lr_decay_ged_sb_5', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.bool.gte(5)
#                 #     .transform.temporal.time_since()
#                 #     .transform.temporal.decay(12)
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('lr_decay_ged_sb_25', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.bool.gte(25)
#                 #     .transform.temporal.time_since()
#                 #     .transform.temporal.decay(12)
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('lr_decay_ged_sb_100', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.bool.gte(100)
#                 #     .transform.temporal.time_since()
#                 #     .transform.temporal.decay(12)
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('lr_decay_ged_sb_500', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.bool.gte(500)
#                 #     .transform.temporal.time_since()
#                 #     .transform.temporal.decay(12)
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('lr_decay_ged_os_1', from_loa='priogrid_month', from_column='ged_os_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.bool.gte(1)
#                 #     .transform.temporal.time_since()
#                 #     .transform.temporal.decay(24)
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('lr_decay_ged_os_5', from_loa='priogrid_month', from_column='ged_os_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.bool.gte(5)
#                 #     .transform.temporal.time_since()
#                 #     .transform.temporal.decay(12)
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('lr_decay_ged_os_25', from_loa='priogrid_month', from_column='ged_os_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.bool.gte(25)
#                 #     .transform.temporal.time_since()
#                 #     .transform.temporal.decay(12)
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('lr_decay_ged_os_100', from_loa='priogrid_month', from_column='ged_os_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.bool.gte(100)
#                 #     .transform.temporal.time_since()
#                 #     .transform.temporal.decay(12)
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('lr_decay_ged_os_500', from_loa='priogrid_month', from_column='ged_os_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.bool.gte(500)
#                 #     .transform.temporal.time_since()
#                 #     .transform.temporal.decay(12)
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('lr_decay_ged_ns_5', from_loa='priogrid_month', from_column='ged_ns_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.bool.gte(5)
#                 #     .transform.temporal.time_since()
#                 #     .transform.temporal.decay(12)
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('lr_decay_ged_ns_1', from_loa='priogrid_month', from_column='ged_ns_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.bool.gte(1)
#                 #     .transform.temporal.time_since()
#                 #     .transform.temporal.decay(24)
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('lr_decay_ged_ns_25', from_loa='priogrid_month', from_column='ged_ns_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.bool.gte(25)
#                 #     .transform.temporal.time_since()
#                 #     .transform.temporal.decay(12)
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('lr_decay_ged_ns_100', from_loa='priogrid_month', from_column='ged_ns_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.bool.gte(100)
#                 #     .transform.temporal.time_since()
#                 #     .transform.temporal.decay(12)
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('lr_decay_ged_ns_500', from_loa='priogrid_month', from_column='ged_ns_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.bool.gte(500)
#                 #     .transform.temporal.time_since()
#                 #     .transform.temporal.decay(12)
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('lr_splag_1_1_sb_1', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.bool.gte(1)
#                 #     .transform.temporal.time_since()
#                 #     .transform.temporal.decay(24)
#                 #     .transform.spatial.lag(1,1,0,0)
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('lr_splag_1_decay_ged_sb_1', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.bool.gte(1)
#                 #     .transform.temporal.time_since()
#                 #     .transform.temporal.decay(24)
#                 #     .transform.spatial.lag(1,1,0,0)
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('lr_splag_1_decay_ged_os_1', from_loa='priogrid_month', from_column='ged_os_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.bool.gte(1)
#                 #     .transform.temporal.time_since()
#                 #     .transform.temporal.decay(24)
#                 #     .transform.spatial.lag(1,1,0,0)
#                 #     .transform.missing.replace_na()
#                 #     )

#                 # .with_column(Column('lr_splag_1_decay_ged_ns_1', from_loa='priogrid_month', from_column='ged_ns_best_sum_nokgi')
#                 #     .transform.missing.replace_na()
#                 #     .transform.bool.gte(1)
#                 #     .transform.temporal.time_since()
#                 #     .transform.temporal.decay(24)
#                 #     .transform.spatial.lag(1,1,0,0)
#                 #     .transform.missing.replace_na()
#                 #     )

#                 .with_theme('fatalities')
#                 .describe("""Fatalities natural and social geography, pgm level

#                                     Predicting fatalities using natural and social geography features

#                                     """)
#                 )
    
#     return qs_natsoc
