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

    queryset = (Queryset('fatalities003_topics','country_month')
        .with_column(Column('ln_ged_sb_dep', from_loa='country_month', from_column='ged_sb_best_sum_nokgi')
            .transform.ops.ln()
            .transform.missing.fill()
            )

        .with_column(Column('ln_ged_sb', from_loa='country_month', from_column='ged_sb_best_sum_nokgi')
            .transform.ops.ln()
            .transform.missing.fill()
            )

        .with_column(Column('lr_wdi_sp_pop_totl', from_loa='country_year', from_column='wdi_sp_pop_totl')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('lr_topic_tokens_t1', from_loa='country_month', from_column='topic_tokens')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(1)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_tokens_t2', from_loa='country_month', from_column='topic_tokens')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(2)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_tokens_t13', from_loa='country_month', from_column='topic_tokens')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(13)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta0_stock_t1', from_loa='country_month', from_column='topic_ste_theta0_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(1)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta0_stock_t2', from_loa='country_month', from_column='topic_ste_theta0_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(2)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta0_stock_t13', from_loa='country_month', from_column='topic_ste_theta0_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(13)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta1_stock_t1', from_loa='country_month', from_column='topic_ste_theta1_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(1)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta1_stock_t2', from_loa='country_month', from_column='topic_ste_theta1_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(2)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta1_stock_t13', from_loa='country_month', from_column='topic_ste_theta1_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(13)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta2_stock_t1', from_loa='country_month', from_column='topic_ste_theta2_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(1)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta2_stock_t2', from_loa='country_month', from_column='topic_ste_theta2_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(2)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta2_stock_t13', from_loa='country_month', from_column='topic_ste_theta2_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(13)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta3_stock_t1', from_loa='country_month', from_column='topic_ste_theta3_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(1)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta3_stock_t2', from_loa='country_month', from_column='topic_ste_theta3_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(2)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta3_stock_t13', from_loa='country_month', from_column='topic_ste_theta3_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(13)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta4_stock_t1', from_loa='country_month', from_column='topic_ste_theta4_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(1)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta4_stock_t2', from_loa='country_month', from_column='topic_ste_theta4_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(2)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta4_stock_t13', from_loa='country_month', from_column='topic_ste_theta4_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(13)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta5_stock_t1', from_loa='country_month', from_column='topic_ste_theta5_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(1)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta5_stock_t2', from_loa='country_month', from_column='topic_ste_theta5_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(2)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta5_stock_t13', from_loa='country_month', from_column='topic_ste_theta5_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(13)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta6_stock_t1', from_loa='country_month', from_column='topic_ste_theta6_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(1)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta6_stock_t2', from_loa='country_month', from_column='topic_ste_theta6_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(2)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta6_stock_t13', from_loa='country_month', from_column='topic_ste_theta6_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(13)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta7_stock_t1', from_loa='country_month', from_column='topic_ste_theta7_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(1)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta7_stock_t2', from_loa='country_month', from_column='topic_ste_theta7_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(2)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta7_stock_t13', from_loa='country_month', from_column='topic_ste_theta7_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(13)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta8_stock_t1', from_loa='country_month', from_column='topic_ste_theta8_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(1)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta8_stock_t2', from_loa='country_month', from_column='topic_ste_theta8_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(2)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta8_stock_t13', from_loa='country_month', from_column='topic_ste_theta8_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(13)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta9_stock_t1', from_loa='country_month', from_column='topic_ste_theta9_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(1)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta9_stock_t2', from_loa='country_month', from_column='topic_ste_theta9_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(2)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta9_stock_t13', from_loa='country_month', from_column='topic_ste_theta9_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(13)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta10_stock_t1', from_loa='country_month', from_column='topic_ste_theta10_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(1)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta10_stock_t2', from_loa='country_month', from_column='topic_ste_theta10_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(2)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta10_stock_t13', from_loa='country_month', from_column='topic_ste_theta10_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(13)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta11_stock_t1', from_loa='country_month', from_column='topic_ste_theta11_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(1)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta11_stock_t2', from_loa='country_month', from_column='topic_ste_theta11_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(2)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta11_stock_t13', from_loa='country_month', from_column='topic_ste_theta11_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(13)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta12_stock_t1', from_loa='country_month', from_column='topic_ste_theta12_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(1)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta12_stock_t2', from_loa='country_month', from_column='topic_ste_theta12_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(2)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta12_stock_t13', from_loa='country_month', from_column='topic_ste_theta12_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(13)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta13_stock_t1', from_loa='country_month', from_column='topic_ste_theta13_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(1)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta13_stock_t2', from_loa='country_month', from_column='topic_ste_theta13_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(2)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta13_stock_t13', from_loa='country_month', from_column='topic_ste_theta13_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(13)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta14_stock_t1', from_loa='country_month', from_column='topic_ste_theta14_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(1)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta14_stock_t2', from_loa='country_month', from_column='topic_ste_theta14_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(2)
            .transform.missing.fill()
            )

        .with_column(Column('lr_topic_ste_theta14_stock_t13', from_loa='country_month', from_column='topic_ste_theta14_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(13)
            .transform.missing.fill()
            )

        .with_column(Column('lr_decay_ged_sb_5', from_loa='country_month', from_column='ged_sb_best_sum_nokgi')
            .transform.missing.replace_na()
            .transform.bool.gte(5)
            .transform.temporal.time_since()
            .transform.temporal.decay(24)
            .transform.missing.replace_na()
            )

        .with_column(Column('lr_decay_ged_os_5', from_loa='country_month', from_column='ged_os_best_sum_nokgi')
            .transform.missing.replace_na()
            .transform.bool.gte(5)
            .transform.temporal.time_since()
            .transform.temporal.decay(24)
            .transform.missing.replace_na()
            )

        .with_column(Column('lr_splag_1_decay_ged_sb_5', from_loa='country_month', from_column='ged_sb_best_sum_nokgi')
            .transform.missing.replace_na()
            .transform.bool.gte(5)
            .transform.temporal.time_since()
            .transform.temporal.decay(24)
            .transform.spatial.countrylag(1,1,0,0)
            .transform.missing.replace_na()
            )

        .with_column(Column('lr_topic_tokens_t1_splag', from_loa='country_month', from_column='topic_tokens')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(13)
            .transform.missing.fill()
            .transform.spatial.countrylag(1,1,0,0)
            .transform.missing.replace_na()
            )

        .with_column(Column('lr_topic_ste_theta0_stock_t1_splag', from_loa='country_month', from_column='topic_ste_theta0_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(13)
            .transform.missing.fill()
            .transform.spatial.countrylag(1,1,0,0)
            .transform.missing.replace_na()
            )

        .with_column(Column('lr_topic_ste_theta1_stock_t1_splag', from_loa='country_month', from_column='topic_ste_theta1_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(13)
            .transform.missing.fill()
            .transform.spatial.countrylag(1,1,0,0)
            .transform.missing.replace_na()
            )

        .with_column(Column('lr_topic_ste_theta2_stock_t1_splag', from_loa='country_month', from_column='topic_ste_theta2_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(13)
            .transform.missing.fill()
            .transform.spatial.countrylag(1,1,0,0)
            .transform.missing.replace_na()
            )

        .with_column(Column('lr_topic_ste_theta3_stock_t1_splag', from_loa='country_month', from_column='topic_ste_theta3_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(13)
            .transform.missing.fill()
            .transform.spatial.countrylag(1,1,0,0)
            .transform.missing.replace_na()
            )

        .with_column(Column('lr_topic_ste_theta4_stock_t1_splag', from_loa='country_month', from_column='topic_ste_theta4_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(13)
            .transform.missing.fill()
            .transform.spatial.countrylag(1,1,0,0)
            .transform.missing.replace_na()
            )

        .with_column(Column('lr_topic_ste_theta5_stock_t1_splag', from_loa='country_month', from_column='topic_ste_theta5_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(13)
            .transform.missing.fill()
            .transform.spatial.countrylag(1,1,0,0)
            .transform.missing.replace_na()
            )

        .with_column(Column('lr_topic_ste_theta6_stock_t1_splag', from_loa='country_month', from_column='topic_ste_theta6_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(13)
            .transform.missing.fill()
            .transform.spatial.countrylag(1,1,0,0)
            .transform.missing.replace_na()
            )

        .with_column(Column('lr_topic_ste_theta7_stock_t1_splag', from_loa='country_month', from_column='topic_ste_theta7_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(13)
            .transform.missing.fill()
            .transform.spatial.countrylag(1,1,0,0)
            .transform.missing.replace_na()
            )

        .with_column(Column('lr_topic_ste_theta8_stock_t1_splag', from_loa='country_month', from_column='topic_ste_theta8_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(13)
            .transform.missing.fill()
            .transform.spatial.countrylag(1,1,0,0)
            .transform.missing.replace_na()
            )

        .with_column(Column('lr_topic_ste_theta9_stock_t1_splag', from_loa='country_month', from_column='topic_ste_theta9_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(13)
            .transform.missing.fill()
            .transform.spatial.countrylag(1,1,0,0)
            .transform.missing.replace_na()
            )

        .with_column(Column('lr_topic_ste_theta10_stock_t1_splag', from_loa='country_month', from_column='topic_ste_theta10_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(13)
            .transform.missing.fill()
            .transform.spatial.countrylag(1,1,0,0)
            .transform.missing.replace_na()
            )

        .with_column(Column('lr_topic_ste_theta11_stock_t1_splag', from_loa='country_month', from_column='topic_ste_theta11_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(13)
            .transform.missing.fill()
            .transform.spatial.countrylag(1,1,0,0)
            .transform.missing.replace_na()
            )

        .with_column(Column('lr_topic_ste_theta12_stock_t1_splag', from_loa='country_month', from_column='topic_ste_theta12_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(13)
            .transform.missing.fill()
            .transform.spatial.countrylag(1,1,0,0)
            .transform.missing.replace_na()
            )

        .with_column(Column('lr_topic_ste_theta13_stock_t1_splag', from_loa='country_month', from_column='topic_ste_theta13_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(13)
            .transform.missing.fill()
            .transform.spatial.countrylag(1,1,0,0)
            .transform.missing.replace_na()
            )

        .with_column(Column('lr_topic_ste_theta14_stock_t1_splag', from_loa='country_month', from_column='topic_ste_theta14_stock')
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.temporal.tlag(13)
            .transform.missing.fill()
            .transform.spatial.countrylag(1,1,0,0)
            .transform.missing.replace_na()
            )

        .with_theme('fatalities002')
        .describe("""Predicting ln(fatalities), cm level
        
                                Queryset with baseline and Mueller & Rauh topic model features
        
                                """)
        )

    return queryset
