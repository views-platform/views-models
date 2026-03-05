from viewser import Queryset, Column
from viewser import Queryset, Column
from views_pipeline_core.managers.model import ModelPathManager

model_name = ModelPathManager.get_model_name_from_path(__file__)

def generate():
    
    qs_natsoc = (Queryset(f'{model_name}','priogrid_month')
                 
                #  .with_column(Column('raw_ged_sb', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi'))

                #  .with_column(Column('raw_ged_os', from_loa='priogrid_month', from_column='ged_os_best_sum_nokgi'))
                 
                # .with_column(Column('lr_ged_sb_dep', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
                #     .transform.missing.replace_na()
                #     # .transform.ops.ln()
                #     )

                .with_column(Column('lr_ged_sb', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
                    # .transform.ops.ln()
                    .transform.missing.fill()
                    )

                .with_column(Column('lr_imr_mean', from_loa='priogrid_year', from_column='imr_mean')
                    .transform.missing.fill()
                    .transform.missing.replace_na()
                    )

                .with_column(Column('lr_mountains_mean', from_loa='priogrid_year', from_column='mountains_mean')
                    .transform.missing.fill()
                    .transform.missing.replace_na()
                    )

                .with_column(Column('lr_dist_diamsec', from_loa='priogrid', from_column='dist_diamsec_s_wgs')
                    .transform.missing.fill()
                    .transform.missing.replace_na()
                    )

                .with_column(Column('lr_dist_petroleum', from_loa='priogrid', from_column='dist_petroleum_s_wgs')
                    .transform.missing.fill()
                    .transform.missing.replace_na()
                    )

                .with_column(Column('lr_agri_ih', from_loa='priogrid_year', from_column='agri_ih')
                    .transform.missing.fill()
                    .transform.missing.replace_na()
                    )

                .with_column(Column('lr_barren_ih', from_loa='priogrid_year', from_column='barren_ih')
                    .transform.missing.fill()
                    .transform.missing.replace_na()
                    )

                .with_column(Column('lr_forest_ih', from_loa='priogrid_year', from_column='forest_ih')
                    .transform.missing.fill()
                    .transform.missing.replace_na()
                    )

                .with_column(Column('lr_pasture_ih', from_loa='priogrid_year', from_column='pasture_ih')
                    .transform.missing.fill()
                    .transform.missing.replace_na()
                    )

                .with_column(Column('lr_savanna_ih', from_loa='priogrid_year', from_column='savanna_ih')
                    .transform.missing.fill()
                    .transform.missing.replace_na()
                    )

                .with_column(Column('lr_shrub_ih', from_loa='priogrid_year', from_column='shrub_ih')
                    .transform.missing.fill()
                    .transform.missing.replace_na()
                    )

                .with_column(Column('lr_urban_ih', from_loa='priogrid_year', from_column='urban_ih')
                    .transform.missing.fill()
                    .transform.missing.replace_na()
                    )

                .with_column(Column('ln_pop_gpw_sum', from_loa='priogrid_year', from_column='pop_gpw_sum')
                    .transform.ops.ln()
                    .transform.missing.fill()
                    .transform.missing.replace_na()
                    )

                .with_column(Column('ln_ttime_mean', from_loa='priogrid_year', from_column='ttime_mean')
                    .transform.ops.ln()
                    .transform.missing.fill()
                    .transform.missing.replace_na()
                    )

                .with_column(Column('ln_gcp_mer', from_loa='priogrid_year', from_column='gcp_mer')
                    .transform.ops.ln()
                    .transform.missing.fill()
                    .transform.missing.replace_na()
                    )

                .with_column(Column('ln_bdist3', from_loa='priogrid_year', from_column='bdist3')
                    .transform.ops.ln()
                    .transform.missing.fill()
                    .transform.missing.replace_na()
                    )

                .with_column(Column('ln_capdist', from_loa='priogrid_year', from_column='capdist')
                    .transform.ops.ln()
                    .transform.missing.fill()
                    .transform.missing.replace_na()
                    )

                .with_column(Column('lr_greq_1_excluded', from_loa='priogrid_year', from_column='excluded')
                    .transform.bool.gte(1)
                    .transform.missing.fill()
                    .transform.missing.replace_na()
                    )

                .with_column(Column('lr_decay_ged_sb_1', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
                    .transform.missing.replace_na()
                    .transform.bool.gte(1)
                    .transform.temporal.time_since()
                    .transform.temporal.decay(24)
                    .transform.missing.replace_na()
                    )

                .with_column(Column('lr_decay_ged_sb_25', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
                    .transform.missing.replace_na()
                    .transform.bool.gte(25)
                    .transform.temporal.time_since()
                    .transform.temporal.decay(24)
                    .transform.missing.replace_na()
                    )

                .with_column(Column('lr_decay_ged_os_1', from_loa='priogrid_month', from_column='ged_os_best_sum_nokgi')
                    .transform.missing.replace_na()
                    .transform.bool.gte(1)
                    .transform.temporal.time_since()
                    .transform.temporal.decay(24)
                    .transform.missing.replace_na()
                    )

                .with_column(Column('lr_splag_1_1_sb_1', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
                    .transform.missing.replace_na()
                    .transform.bool.gte(1)
                    .transform.temporal.time_since()
                    .transform.temporal.decay(24)
                    .transform.spatial.lag(1,1,0,0)
                    .transform.missing.replace_na()
                    )

                .with_column(Column('lr_splag_1_decay_ged_sb_1', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
                    .transform.missing.replace_na()
                    .transform.bool.gte(1)
                    .transform.temporal.time_since()
                    .transform.temporal.decay(24)
                    .transform.spatial.lag(1,1,0,0)
                    .transform.missing.replace_na()
                    )

                .with_theme('fatalities')
                .describe("""Fatalities natural and social geography, pgm level

                                    Predicting fatalities using natural and social geography features

                                    """)
                )
    
    return qs_natsoc