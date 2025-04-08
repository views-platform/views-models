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

    qs_nowcasting_cm = (Queryset("nowcasting", "country_month")

    .with_column(Column('ln_ged_sb_dep', from_loa='country_month', from_column='ged_sb_best_sum_nokgi')
        .transform.ops.ln()
        .transform.missing.fill()
        )
    
    .with_column(Column('country_name', from_loa = 'country', from_column = 'name'))
    .with_column(Column('iso_ab', from_loa = 'country', from_column = 'isoab'))
    .with_column(Column('gleditsch_ward', from_loa = 'country', from_column = 'gwcode'))
    .with_column(Column('year', from_loa = 'month', from_column = 'year_id'))
    .with_column(Column('month', from_loa = 'month', from_column = 'month'))

    # UCDP GED FINAL                    
    .with_column(Column('sb_final_best', from_loa = 'country_month', from_column = 'ged_sb_best_sum_nokgi'))
    .with_column(Column('ns_final_best', from_loa = 'country_month', from_column = 'ged_ns_best_sum_nokgi'))
    .with_column(Column('os_final_best', from_loa = 'country_month', from_column = 'ged_os_best_sum_nokgi'))
    .with_column(Column('sb_ged_count_events', from_loa = 'country_month', from_column = 'ged_sb_best_count_nokgi'))
    .with_column(Column('ns_ged_count_events', from_loa = 'country_month', from_column = 'ged_ns_best_count_nokgi'))
    .with_column(Column('os_ged_count_events', from_loa = 'country_month', from_column = 'ged_os_best_count_nokgi'))

    # Logged VERSIONS
    .with_column(Column("sb_final_best_ln", from_loa = "country_month", from_column = "ged_sb_best_sum_nokgi")
        .transform.ops.ln()
        .transform.missing.replace_na()) 
                    
    .with_column(Column("ns_final_best_ln", from_loa = "country_month", from_column = "ged_ns_best_sum_nokgi")
        .transform.ops.ln()
        .transform.missing.replace_na()) 
                    
    .with_column(Column("os_final_best_ln", from_loa = "country_month", from_column = "ged_os_best_sum_nokgi")
        .transform.ops.ln()
        .transform.missing.replace_na()) 

    # LAGGED VERSIONS
    .with_column(Column("sb_final_best_ln_1", from_loa = "country_month", from_column = "ged_sb_best_sum_nokgi")
        .transform.ops.ln()
        .transform.missing.replace_na()
        .transform.temporal.tlag(1)
        .transform.missing.replace_na()) 

    .with_column(Column("sb_final_best_ln_2", from_loa = "country_month", from_column = "ged_sb_best_sum_nokgi")
        .transform.ops.ln()
        .transform.missing.replace_na()
        .transform.temporal.tlag(2)
        .transform.missing.replace_na())
                    
    .with_column(Column("sb_final_best_ln_3", from_loa = "country_month", from_column = "ged_sb_best_sum_nokgi")
        .transform.ops.ln()
        .transform.missing.replace_na()
        .transform.temporal.tlag(3)
        .transform.missing.replace_na())
                    
    .with_column(Column("sb_final_best_ln_4", from_loa = "country_month", from_column = "ged_sb_best_sum_nokgi")
        .transform.ops.ln()
        .transform.missing.replace_na()
        .transform.temporal.tlag(4)
        .transform.missing.replace_na())
                    
    .with_column(Column("sb_final_best_ln_5", from_loa = "country_month", from_column = "ged_sb_best_sum_nokgi")
        .transform.ops.ln()
        .transform.missing.replace_na()
        .transform.temporal.tlag(5)
        .transform.missing.replace_na())

    .with_column(Column("sb_final_best_ln_6", from_loa = "country_month", from_column = "ged_sb_best_sum_nokgi")
        .transform.ops.ln()
        .transform.missing.replace_na()
        .transform.temporal.tlag(6)
        .transform.missing.replace_na())

    .with_column(Column("sb_final_best_ln_7", from_loa = "country_month", from_column = "ged_sb_best_sum_nokgi")
        .transform.ops.ln()
        .transform.missing.replace_na()
        .transform.temporal.tlag(7)
        .transform.missing.replace_na())

    .with_column(Column("sb_final_best_ln_8", from_loa = "country_month", from_column = "ged_sb_best_sum_nokgi")
        .transform.ops.ln()
        .transform.missing.replace_na()
        .transform.temporal.tlag(8)
        .transform.missing.replace_na())

    .with_column(Column("sb_final_best_ln_9", from_loa = "country_month", from_column = "ged_sb_best_sum_nokgi")
        .transform.ops.ln()
        .transform.missing.replace_na()
        .transform.temporal.tlag(9)
        .transform.missing.replace_na())

    .with_column(Column("sb_final_best_ln_12", from_loa = "country_month", from_column = "ged_sb_best_sum_nokgi")
        .transform.ops.ln()
        .transform.missing.replace_na()
        .transform.temporal.tlag(12)
        .transform.missing.replace_na()) 
                    
    .with_column(Column("sb_final_best_ln_24", from_loa = "country_month", from_column = "ged_sb_best_sum_nokgi")
        .transform.ops.ln()
        .transform.missing.replace_na()
        .transform.temporal.tlag(24)
        .transform.missing.replace_na())  
                    
    # ACLED DATA                    
    .with_column(Column('acled_sb_fat', from_loa = 'country_month', from_column = 'acled_sb_fat'))
                    
    .with_column(Column('acled_sb_fat_ln', from_loa = 'country_month', from_column = 'acled_sb_fat')
        .transform.ops.ln()
        .transform.missing.fill())
                    
    .with_column(Column('acled_ns_fat_ln', from_loa = 'country_month', from_column = 'acled_ns_fat')
        .transform.ops.ln()
        .transform.missing.fill())
                    
    .with_column(Column('acled_os_fat_ln', from_loa = 'country_month', from_column = 'acled_os_fat')
        .transform.ops.ln()
        .transform.missing.fill())
                    
    .with_column(Column('acled_sb_fat_ln_1', from_loa = 'country_month', from_column = 'acled_sb_fat')
        .transform.ops.ln()
        .transform.missing.replace_na()
        .transform.temporal.tlag(1)
        .transform.missing.replace_na())
                        
    .with_column(Column('acled_sb_fat_ln_2', from_loa = 'country_month', from_column = 'acled_sb_fat')
        .transform.ops.ln()
        .transform.missing.replace_na()
        .transform.temporal.tlag(2)
        .transform.missing.replace_na()) 
                    
    .with_column(Column('acled_sb_fat_ln_3', from_loa = 'country_month', from_column = 'acled_sb_fat')
        .transform.ops.ln()
        .transform.missing.replace_na()
        .transform.temporal.tlag(3)
        .transform.missing.replace_na()) 

    .with_column(Column('acled_sb_fat_ln_12', from_loa = 'country_month', from_column = 'acled_sb_fat')
        .transform.ops.ln()
        .transform.missing.replace_na()
        .transform.temporal.tlag(12)
        .transform.missing.replace_na()) 

    .with_column(Column('acled_sb_fat_ln_24', from_loa = 'country_month', from_column = 'acled_sb_fat')
        .transform.ops.ln()
        .transform.missing.replace_na()
        .transform.temporal.tlag(24)
        .transform.missing.replace_na())

    .with_column(Column('acled_sb_count', from_loa = 'country_month', from_column = 'acled_sb_count'))
    .with_column(Column('acled_ns_fat', from_loa = 'country_month', from_column = 'acled_ns_fat'))
    .with_column(Column('acled_ns_count', from_loa = 'country_month', from_column = 'acled_ns_count'))
    .with_column(Column('acled_os_fat', from_loa = 'country_month', from_column = 'acled_os_fat'))
    .with_column(Column('acled_os_count', from_loa = 'country_month', from_column = 'acled_os_count'))
    .with_column(Column('acled_pr_count', from_loa = 'country_month', from_column = 'acled_pr_count'))


    .with_column(Column("ln_acled_sb_reb", from_loa="country_month", from_column="acled_bat_reb_fat")
        .transform.ops.ln()
        .transform.missing.fill())

    .with_column(Column("ln_acled_sb_gov", from_loa="country_month", from_column="acled_bat_gov_fat")
        .transform.ops.ln()
        .transform.missing.fill())

    # TOPICS DATA                                    
    .with_column(Column('topic_conflict_1', from_loa = 'country_month', from_column = 'topic_ste_theta6')
        .transform.missing.fill()
        .transform.missing.replace_na()
        .transform.temporal.tlag(1)
        .transform.missing.fill())

    .with_column(Column('topic_judiciary_1', from_loa = 'country_month', from_column = 'topic_ste_theta5')
        .transform.missing.fill()
        .transform.missing.replace_na()
        .transform.temporal.tlag(1)
        .transform.missing.fill())

    .with_column(Column('topic_diplomacy_1', from_loa = 'country_month', from_column = 'topic_ste_theta2')
        .transform.missing.fill()
        .transform.missing.replace_na()
        .transform.temporal.tlag(1)
        .transform.missing.fill())
                    
    # VDEM Data
    .with_column(Column("vdem_v2x_delibdem", from_loa="country_year", from_column="vdem_v2x_delibdem")
        .transform.missing.fill()
        .transform.temporal.tlag(12)
        .transform.missing.fill()
        .transform.missing.replace_na())

    .with_column(Column("vdem_v2x_clphy", from_loa="country_year", from_column="vdem_v2x_clphy")
        .transform.missing.fill()
        .transform.temporal.tlag(12)
        .transform.missing.fill()
        .transform.missing.replace_na())

    .with_column(Column("vdem_v2x_rule", from_loa="country_year", from_column="vdem_v2x_rule")
        .transform.missing.fill()
        .transform.temporal.tlag(12)
        .transform.missing.fill()
        .transform.missing.replace_na())
                    
    .with_column(Column("vdem_v2x_freexp", from_loa="country_year", from_column="vdem_v2x_freexp")
        .transform.missing.fill()
        .transform.temporal.tlag(12)
        .transform.missing.fill()
        .transform.missing.replace_na())

    # Topic Models

    .with_column(Column('topic_tokens_t1', from_loa='country_month', from_column='topic_tokens')
        .transform.missing.fill()
        .transform.missing.replace_na()
        .transform.temporal.tlag(1)
        .transform.missing.fill()
        .transform.missing.replace_na()
            )

    .with_column(Column('topic_tokens_t2', from_loa='country_month', from_column='topic_tokens')
        .transform.missing.fill()
        .transform.missing.replace_na()
        .transform.temporal.tlag(2)
        .transform.missing.fill()
        .transform.missing.replace_na())

    .with_column(Column('topic_ste_theta4_stock_t1', from_loa='country_month', from_column='topic_ste_theta4_stock')
        .transform.missing.fill()
        .transform.missing.replace_na()
        .transform.temporal.tlag(1)
        .transform.missing.fill()
        .transform.missing.replace_na())

    .with_column(Column('topic_ste_theta4_stock_t2', from_loa='country_month', from_column='topic_ste_theta4_stock')
        .transform.missing.fill()
        .transform.missing.replace_na()
        .transform.temporal.tlag(2)
        .transform.missing.fill()
        .transform.missing.replace_na())

    .with_column(Column('topic_ste_theta4_stock_t13', from_loa='country_month', from_column='topic_ste_theta4_stock')
        .transform.missing.fill()
        .transform.missing.replace_na()
        .transform.temporal.tlag(13)
        .transform.missing.fill()
        .transform.missing.replace_na())

    .with_column(Column('topic_ste_theta2_stock_t1', from_loa='country_month', from_column='topic_ste_theta5_stock')
        .transform.missing.fill()
        .transform.missing.replace_na()
        .transform.temporal.tlag(1)
        .transform.missing.fill()
        .transform.missing.replace_na())

    .with_column(Column('topic_ste_theta2_stock_t2', from_loa='country_month', from_column='topic_ste_theta5_stock')
        .transform.missing.fill()
        .transform.missing.replace_na()
        .transform.temporal.tlag(2)
        .transform.missing.fill()
        .transform.missing.replace_na())

    .with_column(Column('topic_ste_theta2_stock_t13', from_loa='country_month', from_column='topic_ste_theta5_stock')
        .transform.missing.fill()
        .transform.missing.replace_na()
        .transform.temporal.tlag(13)
        .transform.missing.fill()
        .transform.missing.replace_na())

    # Nowcasting Data
    # UCDP GED FINAL                    
    .with_column(Column('sb_candidate_best', from_loa = 'country_month', from_column = 'candidate_sb_best'))
    .with_column(Column('ns_candidate_best', from_loa = 'country_month', from_column = 'candidate_ns_best'))
    .with_column(Column('os_candidate_best', from_loa = 'country_month', from_column = 'candidate_os_best'))

    .with_column(Column('sb_candidate_high', from_loa = 'country_month', from_column = 'candidate_sb_high'))
    .with_column(Column('ns_candidate_high', from_loa = 'country_month', from_column = 'candidate_ns_high'))
    .with_column(Column('os_candidate_high', from_loa = 'country_month', from_column = 'candidate_os_high'))

    .with_column(Column('sb_candidate_count_events', from_loa = 'country_month', from_column = 'candidate_sb_count'))
    .with_column(Column('ns_candidate_count_events', from_loa = 'country_month', from_column = 'candidate_ns_count'))
    .with_column(Column('os_candidate_count_events', from_loa = 'country_month', from_column = 'candidate_os_count'))

    .with_column(Column('sb_candidate_high_count', from_loa = 'country_month', from_column = 'candidate_sb_high_count'))
    .with_column(Column('ns_candidate_high_count', from_loa = 'country_month', from_column = 'candidate_ns_high_count'))
    .with_column(Column('os_candidate_high_count', from_loa = 'country_month', from_column = 'candidate_os_high_count'))

    # Logged VERSIONS
    .with_column(Column('sb_candidate_best_ln', from_loa='country_month', from_column='candidate_sb_best')
        .transform.ops.ln()
        .transform.missing.replace_na())

    .with_column(Column('ns_candidate_best_ln', from_loa='country_month', from_column='candidate_ns_best')
        .transform.ops.ln()
        .transform.missing.replace_na())

    .with_column(Column('os_candidate_best_ln', from_loa='country_month', from_column='candidate_os_best')
        .transform.ops.ln()
        .transform.missing.replace_na())

    .with_column(Column('sb_candidate_high_ln', from_loa='country_month', from_column='candidate_sb_high')
        .transform.ops.ln()
        .transform.missing.replace_na())

    .with_column(Column('ns_candidate_high_ln', from_loa='country_month', from_column='candidate_ns_high')
        .transform.ops.ln()
        .transform.missing.replace_na())

    .with_column(Column('os_candidate_high_ln', from_loa='country_month', from_column='candidate_os_high')
        .transform.ops.ln()
        .transform.missing.replace_na())

    .with_column(Column('sb_candidate_count_events_ln', from_loa='country_month', from_column='candidate_sb_count')
        .transform.ops.ln()
        .transform.missing.replace_na())

    .with_column(Column('ns_candidate_count_events_ln', from_loa='country_month', from_column='candidate_ns_count')
        .transform.ops.ln()
        .transform.missing.replace_na())

    .with_column(Column('os_candidate_count_events_ln', from_loa='country_month', from_column='candidate_os_count')
        .transform.ops.ln()
        .transform.missing.replace_na())

    .with_column(Column('sb_candidate_high_count_ln', from_loa='country_month', from_column='candidate_sb_high_count')
        .transform.ops.ln()
        .transform.missing.replace_na())

    .with_column(Column('ns_candidate_high_count_ln', from_loa='country_month', from_column='candidate_ns_high_count')
        .transform.ops.ln()
        .transform.missing.replace_na())

    .with_column(Column('os_candidate_high_count_ln', from_loa='country_month', from_column='candidate_os_high_count')
        .transform.ops.ln()
        .transform.missing.replace_na())

    
                        
    # Lagged versions 
    .with_column(Column("sb_candidate_best_ln_1", from_loa = "country_month", from_column = "candidate_sb_best")
        .transform.ops.ln()
        .transform.missing.replace_na()
        .transform.temporal.tlag(1)
        .transform.missing.replace_na()) 

    .with_column(Column("sb_candidate_best_ln_2", from_loa = "country_month", from_column = "candidate_sb_best")
        .transform.ops.ln()
        .transform.missing.replace_na()
        .transform.temporal.tlag(2)
        .transform.missing.replace_na())
                    
    .with_column(Column("sb_candidate_best_ln_3", from_loa = "country_month", from_column = "candidate_sb_best")
        .transform.ops.ln()
        .transform.missing.replace_na()
        .transform.temporal.tlag(3)
        .transform.missing.replace_na())
                    
    .with_column(Column("sb_candidate_best_ln_4", from_loa = "country_month", from_column = "candidate_sb_best")
        .transform.ops.ln()
        .transform.missing.replace_na()
        .transform.temporal.tlag(4)
        .transform.missing.replace_na())
                    
    .with_column(Column("sb_candidate_best_ln_5", from_loa = "country_month", from_column = "candidate_sb_best")
        .transform.ops.ln()
        .transform.missing.replace_na()
        .transform.temporal.tlag(5)
        .transform.missing.replace_na())

    .with_column(Column("sb_candidate_best_ln_6", from_loa = "country_month", from_column = "candidate_sb_best")
        .transform.ops.ln()
        .transform.missing.replace_na()
        .transform.temporal.tlag(6)
        .transform.missing.replace_na())

    ## High
    .with_column(Column("sb_candidate_high_ln_1", from_loa = "country_month", from_column = "candidate_sb_high")
        .transform.ops.ln()
        .transform.missing.replace_na()
        .transform.temporal.tlag(1)
        .transform.missing.replace_na()) 

    .with_column(Column("sb_candidate_high_ln_2", from_loa = "country_month", from_column = "candidate_sb_high")
        .transform.ops.ln()
        .transform.missing.replace_na()
        .transform.temporal.tlag(2)
        .transform.missing.replace_na())
                    
    .with_column(Column("sb_candidate_high_ln_3", from_loa = "country_month", from_column = "candidate_sb_high")
        .transform.ops.ln()
        .transform.missing.replace_na()
        .transform.temporal.tlag(3)
        .transform.missing.replace_na())
                    
    .with_column(Column("sb_candidate_high_ln_4", from_loa = "country_month", from_column = "candidate_sb_high")
        .transform.ops.ln()
        .transform.missing.replace_na()
        .transform.temporal.tlag(4)
        .transform.missing.replace_na())
                    
    .with_column(Column("sb_candidate_high_ln_5", from_loa = "country_month", from_column = "candidate_sb_high")
        .transform.ops.ln()
        .transform.missing.replace_na()
        .transform.temporal.tlag(5)
        .transform.missing.replace_na())

    .with_column(Column("sb_candidate_high_ln_6", from_loa = "country_month", from_column = "candidate_sb_high")
        .transform.ops.ln()
        .transform.missing.replace_na()
        .transform.temporal.tlag(6)
        .transform.missing.replace_na()))

    .with_theme('fatalities002')
    .describe("""Predicting ln(fatalities), cm level for Nowcasting""")

    )

    return queryset_base
