from viewser import Queryset, Column
from views_pipeline_core.managers.model import ModelPathManager

model_name = ModelPathManager.get_model_name_from_path(__file__)


def generate():
    
    qs_conflictlong = (Queryset(f'{model_name}','priogrid_month')
                       

        .with_column(Column('raw_ged_sb_dep', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi'))

        .with_column(Column('raw_ged_os', from_loa='priogrid_month', from_column='ged_os_best_sum_nokgi'))

        .with_column(Column('raw_ged_ns', from_loa='priogrid_month', from_column='ged_ns_best_sum_nokgi'))

        .with_column(Column('ln_ged_sb_dep', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
            .transform.missing.replace_na()
            .transform.ops.ln()
            )

        .with_column(Column('lr_ged_sb', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('lr_ged_os', from_loa='priogrid_month', from_column='ged_os_best_sum_nokgi')
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('lr_ged_ns', from_loa='priogrid_month', from_column='ged_ns_best_sum_nokgi')
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('ln_pop_gpw_sum', from_loa='priogrid_year', from_column='pop_gpw_sum')
            .transform.ops.ln()
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('lr_decay_ged_sb_5', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
            .transform.missing.replace_na()
            .transform.bool.gte(5)
            .transform.temporal.time_since()
            .transform.temporal.decay(12)
            .transform.missing.replace_na()
            )

        .with_column(Column('lr_decay_ged_sb_25', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
            .transform.missing.replace_na()
            .transform.bool.gte(25)
            .transform.temporal.time_since()
            .transform.temporal.decay(12)
            .transform.missing.replace_na()
            )

        .with_column(Column('lr_decay_ged_sb_100', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
            .transform.missing.replace_na()
            .transform.bool.gte(100)
            .transform.temporal.time_since()
            .transform.temporal.decay(12)
            .transform.missing.replace_na()
            )

        .with_column(Column('lr_decay_ged_sb_500', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
            .transform.missing.replace_na()
            .transform.bool.gte(500)
            .transform.temporal.time_since()
            .transform.temporal.decay(12)
            .transform.missing.replace_na()
            )

        .with_column(Column('lr_decay_ged_os_5', from_loa='priogrid_month', from_column='ged_os_best_sum_nokgi')
            .transform.missing.replace_na()
            .transform.bool.gte(5)
            .transform.temporal.time_since()
            .transform.temporal.decay(12)
            .transform.missing.replace_na()
            )

        .with_column(Column('lr_decay_ged_os_25', from_loa='priogrid_month', from_column='ged_os_best_sum_nokgi')
            .transform.missing.replace_na()
            .transform.bool.gte(25)
            .transform.temporal.time_since()
            .transform.temporal.decay(12)
            .transform.missing.replace_na()
            )

        .with_column(Column('lr_decay_ged_os_100', from_loa='priogrid_month', from_column='ged_os_best_sum_nokgi')
            .transform.missing.replace_na()
            .transform.bool.gte(100)
            .transform.temporal.time_since()
            .transform.temporal.decay(12)
            .transform.missing.replace_na()
            )

        .with_column(Column('lr_decay_ged_os_500', from_loa='priogrid_month', from_column='ged_os_best_sum_nokgi')
            .transform.missing.replace_na()
            .transform.bool.gte(500)
            .transform.temporal.time_since()
            .transform.temporal.decay(12)
            .transform.missing.replace_na()
            )

        .with_column(Column('lr_decay_ged_ns_5', from_loa='priogrid_month', from_column='ged_ns_best_sum_nokgi')
            .transform.missing.replace_na()
            .transform.bool.gte(5)
            .transform.temporal.time_since()
            .transform.temporal.decay(12)
            .transform.missing.replace_na()
            )

        .with_column(Column('lr_decay_ged_ns_25', from_loa='priogrid_month', from_column='ged_ns_best_sum_nokgi')
            .transform.missing.replace_na()
            .transform.bool.gte(25)
            .transform.temporal.time_since()
            .transform.temporal.decay(12)
            .transform.missing.replace_na()
            )

        .with_column(Column('lr_decay_ged_ns_100', from_loa='priogrid_month', from_column='ged_ns_best_sum_nokgi')
            .transform.missing.replace_na()
            .transform.bool.gte(100)
            .transform.temporal.time_since()
            .transform.temporal.decay(12)
            .transform.missing.replace_na()
            )

        .with_column(Column('lr_decay_ged_ns_500', from_loa='priogrid_month', from_column='ged_ns_best_sum_nokgi')
            .transform.missing.replace_na()
            .transform.bool.gte(500)
            .transform.temporal.time_since()
            .transform.temporal.decay(12)
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
        .describe("""fatalities longer conflict history, pgm level

                                Predicting ln(ged_best_sb) using conflict predictors, longer version

                                """)
    )
    return qs_conflictlong