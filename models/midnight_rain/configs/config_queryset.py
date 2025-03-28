from viewser import Queryset, Column

def generate():
    """
    Contains the configuration for the input data in the form of a viewser queryset. That is the data from viewser that is used to train the model.
    This configuration is "behavioral" so modifying it will affect the model's runtime behavior and integration into the deployment system.
    There is no guarantee that the model will work if the input data configuration is changed here without changing the model settings and algorithm accordingly.

    Returns:
    - queryset_base (Queryset): A queryset containing the base data for the model training.
    """
    
    qs_escwa_drought = (Queryset('fatalities003_pgm_escwa_drought','priogrid_month')
                        
              .with_column(Column('lr_pgd_nlights_calib_mean', from_loa='priogrid_year', from_column='nlights_calib_mean')
                     .transform.missing.replace_na(0)
                     )

              .with_column(Column('lr_pgd_imr_mean', from_loa='priogrid_year', from_column='imr_mean')
                     .transform.missing.replace_na(0)
                     )

              .with_column(Column('lr_pgd_urban_ih', from_loa='priogrid_year', from_column='urban_ih')
                     .transform.missing.replace_na(0)
                     )

              .with_column(Column('lr_count_moder_drought_prev10', from_loa='priogrid_month', from_column='count_moder_drought_prev10')
                     .transform.missing.replace_na(0)
                     )

              .with_column(Column('lr_cropprop', from_loa='priogrid_month', from_column='cropprop')
                     .transform.missing.replace_na(0)
                     )

              .with_column(Column('lr_growseasdummy', from_loa='priogrid_month', from_column='growseasdummy')
                     .transform.missing.replace_na(0)
                     )

              .with_column(Column('lr_spei1_gs_prev10', from_loa='priogrid_month', from_column='spei1_gs_prev10')
                     .transform.missing.replace_na(0)
                     )

              .with_column(Column('lr_spei1_gs_prev10_anom', from_loa='priogrid_month', from_column='spei1_gs_prev10_anom')
                     .transform.missing.replace_na(0)
                     )

              .with_column(Column('lr_spei1_gsm_cv_anom', from_loa='priogrid_month', from_column='spei1_gsm_cv_anom')
                     .transform.missing.replace_na(0)
                     )

              .with_column(Column('lr_spei1_gsm_detrend', from_loa='priogrid_month', from_column='spei1_gsm_detrend')
                     .transform.missing.replace_na(0)
                     )

              .with_column(Column('lr_spei1gsy_lowermedian_count', from_loa='priogrid_month', from_column='spei1gsy_lowermedian_count')
                     .transform.missing.replace_na(0)
                     )

              .with_column(Column('lr_spei_48_detrend', from_loa='priogrid_month', from_column='spei_48_detrend')
                     .transform.missing.replace_na(0)
                     )

              .with_column(Column('lr_tlag1_dr_mod_gs', from_loa='priogrid_month', from_column='tlag1_dr_mod_gs')
                     .transform.missing.replace_na(0)
                     )

              .with_column(Column('lr_tlag1_dr_moder_gs', from_loa='priogrid_month', from_column='tlag1_dr_moder_gs')
                     .transform.missing.replace_na(0)
                     )

              .with_column(Column('lr_tlag1_dr_sev_gs', from_loa='priogrid_month', from_column='tlag1_dr_sev_gs')
                     .transform.missing.replace_na(0)
                     )

              .with_column(Column('lr_tlag1_spei1_gsm', from_loa='priogrid_month', from_column='tlag1_spei1_gsm')
                     .transform.missing.replace_na(0)
                     )

              .with_column(Column('lr_tlag_12_crop_sum', from_loa='priogrid_month', from_column='tlag_12_crop_sum')
                     .transform.missing.replace_na(0)
                     )

              .with_column(Column('lr_tlag_12_harvarea_maincrops', from_loa='priogrid_month', from_column='tlag_12_harvarea_maincrops')
                     .transform.missing.replace_na(0)
                     )

              .with_column(Column('lr_tlag_12_irr_maincrops', from_loa='priogrid_month', from_column='tlag_12_irr_maincrops')
                     .transform.missing.replace_na(0)
                     )

              .with_column(Column('lr_tlag_12_rainf_maincrops', from_loa='priogrid_month', from_column='tlag_12_rainf_maincrops')
                     .transform.missing.replace_na(0)
                     )

              .with_column(Column('ln_ged_sb_dep', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
                     .transform.missing.replace_na()
                     .transform.ops.ln()
                     )

              .with_column(Column('ln_ged_sb', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
                     .transform.missing.replace_na()
                     .transform.ops.ln()
                     )

              .with_column(Column('lr_greq_1_excluded', from_loa='priogrid_year', from_column='excluded')
                     .transform.bool.gte(1)
                     .transform.missing.fill()
                     )

              .with_column(Column('ln_pop_gpw_sum', from_loa='priogrid_year', from_column='pop_gpw_sum')
                     .transform.missing.replace_na(0)
                     .transform.ops.ln()
                     )

              .with_column(Column('ln_pgd_ttime_mean', from_loa='priogrid_year', from_column='ttime_mean')
                     .transform.missing.replace_na(0)
                     .transform.ops.ln()
                     )

              .with_column(Column('lr_wdi_nv_agr_totl_kd', from_loa='country_year', from_column='wdi_nv_agr_totl_kd')
                     .transform.missing.replace_na(0)
                     .transform.temporal.tlag(12)
                     .transform.missing.replace_na(0)
                     )

              .with_column(Column('lr_decay_ged_sb_1', from_loa='priogrid_month', from_column='ged_sb_best_sum_nokgi')
                     .transform.missing.replace_na()
                     .transform.bool.gte(1)
                     .transform.temporal.time_since()
                     .transform.temporal.decay(12)
                     .transform.missing.replace_na()
                     )

              .with_column(Column('lr_decay_ged_os_1', from_loa='priogrid_month', from_column='ged_os_best_sum_nokgi')
                     .transform.missing.replace_na()
                     .transform.bool.gte(1)
                     .transform.temporal.time_since()
                     .transform.temporal.decay(12)
                     .transform.missing.replace_na()
                     )

              .with_column(Column('lr_decay_ged_ns_1', from_loa='priogrid_month', from_column='ged_ns_best_sum_nokgi')
                     .transform.missing.replace_na()
                     .transform.bool.gte(1)
                     .transform.temporal.time_since()
                     .transform.temporal.decay(12)
                     .transform.missing.replace_na()
                     )

              .with_theme('fatalities')
              .describe("""Fatalities, escwa drought and vulnerability, pgm level

                                                 Predicting number of fatalities with features from the escwa drought and 
                                                 vulnerability themes

                                   """)
    )
                    
    return qs_escwa_drought