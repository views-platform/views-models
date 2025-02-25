from viewser import Queryset, Column

def generate():
    """
    Contains the configuration for the input data in the form of a viewser queryset. That is the data from viewser that is used to train the model.
    This configuration is "behavioral" so modifying it will affect the model's runtime behavior and integration into the deployment system.
    There is no guarantee that the model will work if the input data configuration is changed here without changing the model settings and algorithm accordingly.

    Returns:
    - queryset_base (Queryset): A queryset containing the base data for the model training.
    """
    
    # VIEWSER 6, Example configuration. Modify as needed.

    queryset = (Queryset('uncertainty_confhist_nolog','country_month')

        .with_column(Column('ged_sb_dep', from_loa='country_month', from_column='ged_sb_best_sum_nokgi')
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('ged_sb', from_loa='country_month', from_column='ged_sb_best_sum_nokgi')
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('ged_ns', from_loa='country_month', from_column='ged_ns_best_sum_nokgi')
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('ged_os', from_loa='country_month', from_column='ged_os_best_sum_nokgi')
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('acled_sb', from_loa='country_month', from_column='acled_sb_fat')
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('ln_acled_sb_count', from_loa='country_month', from_column='acled_sb_count')
            .transform.ops.ln()
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('acled_os', from_loa='country_month', from_column='acled_os_fat')
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('wdi_sp_pop_totl', from_loa='country_year', from_column='wdi_sp_pop_totl')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('ged_sb_tlag_1', from_loa='country_month', from_column='ged_sb_best_sum_nokgi')
            .transform.missing.fill()
            .transform.temporal.tlag(1)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('ged_sb_tlag_2', from_loa='country_month', from_column='ged_sb_best_sum_nokgi')
            .transform.missing.fill()
            .transform.temporal.tlag(2)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('ged_sb_tlag_3', from_loa='country_month', from_column='ged_sb_best_sum_nokgi')
            .transform.missing.fill()
            .transform.temporal.tlag(3)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('ged_sb_tlag_4', from_loa='country_month', from_column='ged_sb_best_sum_nokgi')
            .transform.missing.fill()
            .transform.temporal.tlag(4)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('ged_sb_tlag_5', from_loa='country_month', from_column='ged_sb_best_sum_nokgi')
            .transform.missing.fill()
            .transform.temporal.tlag(5)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('ged_sb_tlag_6', from_loa='country_month', from_column='ged_sb_best_sum_nokgi')
            .transform.missing.fill()
            .transform.temporal.tlag(6)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('ged_os_tlag_1', from_loa='country_month', from_column='ged_os_best_sum_nokgi')
            .transform.missing.fill()
            .transform.temporal.tlag(1)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('decay_ged_sb_5', from_loa='country_month', from_column='ged_sb_best_sum_nokgi')
            .transform.missing.replace_na()
            .transform.bool.gte(5)
            .transform.temporal.time_since()
            .transform.temporal.decay(24)
            .transform.missing.replace_na()
            )

        .with_column(Column('decay_ged_os_5', from_loa='country_month', from_column='ged_os_best_sum_nokgi')
            .transform.missing.replace_na()
            .transform.bool.gte(5)
            .transform.temporal.time_since()
            .transform.temporal.decay(24)
            .transform.missing.replace_na()
            )

        .with_column(Column('decay_ged_sb_100', from_loa='country_month', from_column='ged_sb_best_sum_nokgi')
            .transform.missing.replace_na()
            .transform.bool.gte(100)
            .transform.temporal.time_since()
            .transform.temporal.decay(24)
            .transform.missing.replace_na()
            )

        .with_column(Column('decay_ged_sb_500', from_loa='country_month', from_column='ged_sb_best_sum_nokgi')
            .transform.missing.replace_na()
            .transform.bool.gte(500)
            .transform.temporal.time_since()
            .transform.temporal.decay(24)
            .transform.missing.replace_na()
            )
        .with_column(Column('decay_ged_sb_1000', from_loa='country_month', from_column='ged_sb_best_sum_nokgi')
            .transform.missing.replace_na()
            .transform.bool.gte(1000)
            .transform.temporal.time_since()
            .transform.temporal.decay(24)
            .transform.missing.replace_na()
            )

        .with_column(Column('decay_ged_os_100', from_loa='country_month', from_column='ged_os_best_sum_nokgi')
            .transform.missing.replace_na()
            .transform.bool.gte(100)
            .transform.temporal.time_since()
            .transform.temporal.decay(24)
            .transform.missing.replace_na()
            )

        .with_column(Column('decay_ged_ns_5', from_loa='country_month', from_column='ged_ns_best_sum_nokgi')
            .transform.missing.replace_na()
            .transform.bool.gte(5)
            .transform.temporal.time_since()
            .transform.temporal.decay(24)
            .transform.missing.replace_na()
            )

        .with_column(Column('decay_ged_ns_100', from_loa='country_month', from_column='ged_ns_best_sum_nokgi')
            .transform.missing.replace_na()
            .transform.bool.gte(100)
            .transform.temporal.time_since()
            .transform.temporal.decay(24)
            .transform.missing.replace_na()
            )

        .with_column(Column('decay_acled_sb_5', from_loa='country_month', from_column='acled_sb_fat')
            .transform.missing.replace_na()
            .transform.bool.gte(5)
            .transform.temporal.time_since()
            .transform.temporal.decay(24)
            .transform.missing.replace_na()
            )

        .with_column(Column('decay_acled_os_5', from_loa='country_month', from_column='acled_os_fat')
            .transform.missing.replace_na()
            .transform.bool.gte(5)
            .transform.temporal.time_since()
            .transform.temporal.decay(24)
            .transform.missing.replace_na()
            )

        .with_column(Column('decay_acled_ns_5', from_loa='country_month', from_column='acled_ns_fat')
            .transform.missing.replace_na()
            .transform.bool.gte(5)
            .transform.temporal.time_since()
            .transform.temporal.decay(24)
            .transform.missing.replace_na()
            )

        .with_column(Column('splag_1_decay_ged_sb_100', from_loa='country_month', from_column='ged_sb_best_sum_nokgi')
            .transform.missing.replace_na()
            .transform.bool.gte(100)
            .transform.temporal.time_since()
            .transform.temporal.decay(24)
            .transform.spatial.countrylag(1,1,0,0)
            .transform.missing.replace_na()
            )
        
        .with_column(Column('splag_1_decay_ged_sb_5', from_loa='country_month', from_column='ged_sb_best_sum_nokgi')
            .transform.missing.replace_na()
            .transform.bool.gte(5)
            .transform.temporal.time_since()
            .transform.temporal.decay(24)
            .transform.spatial.countrylag(1,1,0,0)
            .transform.missing.replace_na()
            )

        .with_column(Column('splag_1_decay_ged_os_5', from_loa='country_month', from_column='ged_os_best_sum_nokgi')
            .transform.missing.replace_na()
            .transform.bool.gte(5)
            .transform.temporal.time_since()
            .transform.temporal.decay(24)
            .transform.spatial.countrylag(1,1,0,0)
            .transform.missing.replace_na()
            )

        .with_column(Column('splag_1_decay_ged_ns_5', from_loa='country_month', from_column='ged_ns_best_sum_nokgi')
            .transform.missing.replace_na()
            .transform.bool.gte(5)
            .transform.temporal.time_since()
            .transform.temporal.decay(24)
            .transform.spatial.countrylag(1,1,0,0)
            .transform.missing.replace_na()
            )


        .with_theme('uncertainty')
        .describe("""Predicting ln(fatalities), cm level
        
                                Queryset with conflict history and population data.
        
                                """)
        )

    return queryset