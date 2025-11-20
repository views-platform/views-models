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

    queryset = (Queryset('uncertainty_structural_nolog','country_month')

        .with_column(Column('lr_sb_best', from_loa='country_month', from_column='ged_sb_best_sum_nokgi')
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('wdi_sm_pop_netm', from_loa='country_year', from_column='wdi_sm_pop_netm')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('wdi_sm_pop_refg_or', from_loa='country_year', from_column='wdi_sm_pop_refg_or')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('wdi_dt_oda_odat_pc_zs', from_loa='country_year', from_column='wdi_dt_oda_odat_pc_zs')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('wdi_ms_mil_xpnd_gd_zs', from_loa='country_year', from_column='wdi_ms_mil_xpnd_gd_zs')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('wdi_sl_tlf_totl_fe_zs', from_loa='country_year', from_column='wdi_sl_tlf_totl_fe_zs')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('wdi_nv_agr_totl_kn', from_loa='country_year', from_column='wdi_nv_agr_totl_kn')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('wdi_sp_pop_grow', from_loa='country_year', from_column='wdi_sp_pop_grow')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('wdi_se_enr_prim_fm_zs', from_loa='country_year', from_column='wdi_se_enr_prim_fm_zs')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('wdi_sp_urb_totl_in_zs', from_loa='country_year', from_column='wdi_sp_urb_totl_in_zs')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('wdi_sh_sta_maln_zs', from_loa='country_year', from_column='wdi_sh_sta_maln_zs')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('wdi_sp_dyn_imrt_fe_in', from_loa='country_year', from_column='wdi_sp_dyn_imrt_fe_in')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('wdi_ny_gdp_mktp_kd', from_loa='country_year', from_column='wdi_ny_gdp_mktp_kd')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('wdi_sh_sta_stnt_zs', from_loa='country_year', from_column='wdi_sh_sta_stnt_zs')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('vdem_v2x_horacc', from_loa='country_year', from_column='vdem_v2x_horacc')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('vdem_v2xnp_client', from_loa='country_year', from_column='vdem_v2xnp_client')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('vdem_v2x_veracc', from_loa='country_year', from_column='vdem_v2x_veracc')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('vdem_v2x_divparctrl', from_loa='country_year', from_column='vdem_v2x_divparctrl')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('vdem_v2xpe_exlpol', from_loa='country_year', from_column='vdem_v2xpe_exlpol')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('vdem_v2x_diagacc', from_loa='country_year', from_column='vdem_v2x_diagacc')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('vdem_v2xpe_exlgeo', from_loa='country_year', from_column='vdem_v2xpe_exlgeo')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('vdem_v2xpe_exlgender', from_loa='country_year', from_column='vdem_v2xpe_exlgender')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('vdem_v2xpe_exlsocgr', from_loa='country_year', from_column='vdem_v2xpe_exlsocgr')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('vdem_v2x_ex_party', from_loa='country_year', from_column='vdem_v2x_ex_party')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('vdem_v2x_genpp', from_loa='country_year', from_column='vdem_v2x_genpp')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('vdem_v2xeg_eqdr', from_loa='country_year', from_column='vdem_v2xeg_eqdr')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('vdem_v2xcl_prpty', from_loa='country_year', from_column='vdem_v2xcl_prpty')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('vdem_v2xeg_eqprotec', from_loa='country_year', from_column='vdem_v2xeg_eqprotec')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('vdem_v2x_ex_military', from_loa='country_year', from_column='vdem_v2x_ex_military')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('vdem_v2xcl_dmove', from_loa='country_year', from_column='vdem_v2xcl_dmove')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('vdem_v2x_clphy', from_loa='country_year', from_column='vdem_v2x_clphy')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('vdem_v2x_hosabort', from_loa='country_year', from_column='vdem_v2x_hosabort')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('vdem_v2xnp_regcorr', from_loa='country_year', from_column='vdem_v2xnp_regcorr')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('wdi_sp_pop_totl', from_loa='country_year', from_column='wdi_sp_pop_totl')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

        .with_column(Column('splag_wdi_sl_tlf_totl_fe_zs', from_loa='country_year', from_column='wdi_sl_tlf_totl_fe_zs')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.spatial.countrylag(1,1,0,0)
            .transform.missing.replace_na()
            )

        .with_column(Column('splag_wdi_sm_pop_refg_or', from_loa='country_year', from_column='wdi_sm_pop_refg_or')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.spatial.countrylag(1,1,0,0)
            .transform.missing.replace_na()
            )

        .with_column(Column('splag_wdi_sm_pop_netm', from_loa='country_year', from_column='wdi_sm_pop_netm')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.spatial.countrylag(1,1,0,0)
            .transform.missing.replace_na()
            )

        .with_column(Column('splag_wdi_ag_lnd_frst_k2', from_loa='country_year', from_column='wdi_ag_lnd_frst_k2')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.spatial.countrylag(1,1,0,0)
            .transform.missing.replace_na()
            )

        .with_column(Column('splag_vdem_v2x_libdem', from_loa='country_year', from_column='vdem_v2x_libdem')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.spatial.countrylag(1,1,0,0)
            .transform.missing.replace_na()
            )

        .with_column(Column('splag_vdem_v2x_accountability', from_loa='country_year', from_column='vdem_v2x_accountability')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.spatial.countrylag(1,1,0,0)
            .transform.missing.replace_na()
            )

        .with_column(Column('splag_vdem_v2xpe_exlsocgr', from_loa='country_year', from_column='vdem_v2xpe_exlsocgr')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.spatial.countrylag(1,1,0,0)
            .transform.missing.replace_na()
            )

        .with_column(Column('splag_vdem_v2xcl_rol', from_loa='country_year', from_column='vdem_v2xcl_rol')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.spatial.countrylag(1,1,0,0)
            .transform.missing.replace_na()
            )


        .with_theme('uncertainty')
        .describe("""Predicting ln(fatalities), cm level
        
                                Queryset with baseline and broad list of features from all sources
        
                                """)
        )

    return queryset