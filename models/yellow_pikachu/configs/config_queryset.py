from viewser import Queryset, Column
from views_pipeline_core.managers.model import ModelPathManager

model_name = ModelPathManager.get_model_name_from_path(__file__)


def generate():

    qs_treelag = (Queryset(f'{model_name}','priogrid_month')
                  

                .with_column(Column('raw_ged_sb', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi'))

                .with_column(Column('raw_ged_ns', from_loa='priogrid_month', from_column='ged_ns_best_sum_nokgi'))

                .with_column(Column('raw_ged_os', from_loa='priogrid_month', from_column='ged_os_best_sum_nokgi'))
                  
                .with_column(Column('lr_ged_gte_1', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
                    .transform.bool.gte(1)
                    )

                .with_column(Column('ln_ged_sb_dep', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
                    .transform.missing.replace_na()
                    .transform.ops.ln()
                    )

                .with_column(Column('lr_treelag_1_sb', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
                    .transform.missing.replace_na()
                    .transform.spatial.treelag(0.7,1)
                    )

                .with_column(Column('lr_treelag_1_ns', from_loa='priogrid_month', from_column='ged_ns_best_sum_nokgi')
                    .transform.missing.replace_na()
                    .transform.spatial.treelag(0.7,1)
                    )

                .with_column(Column('lr_treelag_1_os', from_loa='priogrid_month', from_column='ged_os_best_sum_nokgi')
                    .transform.missing.replace_na()
                    .transform.spatial.treelag(0.7,1)
                    )

                .with_column(Column('lr_treelag_2_sb', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
                    .transform.missing.replace_na()
                    .transform.spatial.treelag(0.7,2)
                    )

                .with_column(Column('lr_treelag_2_ns', from_loa='priogrid_month', from_column='ged_ns_best_sum_nokgi')
                    .transform.missing.replace_na()
                    .transform.spatial.treelag(0.7,2)
                    )

                .with_column(Column('lr_treelag_2_os', from_loa='priogrid_month', from_column='ged_os_best_sum_nokgi')
                    .transform.missing.replace_na()
                    .transform.spatial.treelag(0.7,2)
                    )

                )
    return qs_treelag