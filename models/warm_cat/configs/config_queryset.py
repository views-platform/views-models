from viewser import Queryset, Column
from views_pipeline_core.managers.model import ModelPathManager

model_name = ModelPathManager.get_model_name_from_path(__file__)

def generate():
    

    qs_sptime_dist = (Queryset(f'{model_name}','priogrid_month')
                     .with_column(Column('lr_ged_sb', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
                            .transform.missing.replace_na()
                            )

                    .with_column(Column('lr_ged_sb_splag_1', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
                            .transform.spatial.lag(1,1,0,0)
                            .transform.missing.fill()
                            )

                    .with_column(Column('lr_ged_ns_splag_1', from_loa='priogrid_month', from_column='ged_ns_best_count_nokgi')
                            .transform.spatial.lag(1,1,0,0)
                            .transform.missing.fill()
                            )

                    .with_column(Column('lr_ged_os_splag_1', from_loa='priogrid_month', from_column='ged_os_best_count_nokgi')
                            .transform.spatial.lag(1,1,0,0)
                            .transform.missing.fill()
                            )

                     .with_column(Column('lr_sptime_dist_k1_ged_sb', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
                            .transform.missing.replace_na()
                            .transform.spatial.sptime_dist("distances",1,1.0,0.0)
                            )

                     .with_column(Column('lr_sptime_dist_k1_ged_os', from_loa='priogrid_month', from_column='ged_os_best_sum_nokgi')
                            .transform.missing.replace_na()
                            .transform.spatial.sptime_dist("distances",1,1.0,0.0)
                            )

                     .with_column(Column('lr_sptime_dist_k1_ged_ns', from_loa='priogrid_month', from_column='ged_ns_best_sum_nokgi')
                            .transform.missing.replace_na()
                            .transform.spatial.sptime_dist("distances",1,1.0,0.0)
                            )

                     .with_column(Column('lr_sptime_dist_k10_ged_sb', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
                            .transform.missing.replace_na()
                            .transform.spatial.sptime_dist("distances",1,10.0,0.0)
                            )

                     .with_column(Column('lr_sptime_dist_k10_ged_os', from_loa='priogrid_month', from_column='ged_os_best_sum_nokgi')
                            .transform.missing.replace_na()
                            .transform.spatial.sptime_dist("distances",1,10.0,0.0)
                            )

                     .with_column(Column('lr_sptime_dist_k10_ged_ns', from_loa='priogrid_month', from_column='ged_ns_best_sum_nokgi')
                            .transform.missing.replace_na()
                            .transform.spatial.sptime_dist("distances",1,10.0,0.0)
                            )

                     .with_column(Column('lr_sptime_dist_k001_ged_sb', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
                            .transform.missing.replace_na()
                            .transform.spatial.sptime_dist("distances",1,0.01,0.0)
                            )

                     .with_column(Column('lr_sptime_dist_k001_ged_os', from_loa='priogrid_month', from_column='ged_os_best_sum_nokgi')
                            .transform.missing.replace_na()
                            .transform.spatial.sptime_dist("distances",1,0.01,0.0)
                            )

                     .with_column(Column('lr_sptime_dist_k001_ged_ns', from_loa='priogrid_month', from_column='ged_ns_best_sum_nokgi')
                            .transform.missing.replace_na()
                            .transform.spatial.sptime_dist("distances",1,0.01,0.0)
                            )
                    .with_column(Column('lr_dist_diamsec', from_loa='priogrid', from_column='dist_diamsec_s_wgs')
                     .transform.missing.fill()
                     .transform.missing.replace_na()
                     )

                    .with_column(Column('lr_imr_mean', from_loa='priogrid_year', from_column='imr_mean')
                            .transform.missing.fill()
                            .transform.missing.replace_na()
                            )

                    .with_column(Column('lr_ttime_mean', from_loa='priogrid_year', from_column='ttime_mean')
                            .transform.missing.fill()
                            .transform.missing.replace_na()
                            )

                    .with_column(Column('lr_bdist3', from_loa='priogrid_year', from_column='bdist3')
                            .transform.missing.fill()
                            .transform.missing.replace_na()
                            )

                    .with_column(Column('lr_capdist', from_loa='priogrid_year', from_column='capdist')
                            .transform.missing.fill()
                            .transform.missing.replace_na()
                            )

                    .with_column(Column('lr_pop_gpw_sum', from_loa='priogrid_year', from_column='pop_gpw_sum')
                            .transform.missing.fill()
                            .transform.missing.replace_na()
                            )
                     )

    return qs_sptime_dist