from viewser import Queryset, Column

def generate():
    
    qs_baseline = (Queryset('fatalities003_pgm_baseline','priogrid_month')
                   
                .with_column(Column('ln_ged_sb_dep', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
                    .transform.missing.replace_na()
                    .transform.ops.ln()
                    )

                .with_column(Column('ln_ged_sb', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
                    .transform.ops.ln()
                    .transform.missing.fill()
                    )

                .with_column(Column('ln_pop_gpw_sum', from_loa='priogrid_year', from_column='pop_gpw_sum')
                    .transform.ops.ln()
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
                .describe("""Fatalities conflict history, cm level

                                    Predicting ln(fatalities) using conflict predictors, ultrashort

                                    """)
                )
    
    return qs_baseline