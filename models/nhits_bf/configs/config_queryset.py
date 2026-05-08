from viewser import Queryset, Column

def generate():
    queryset = (Queryset("nhits_bf","country_month")

    #---------------------------------
    # Economy
    #---------------------------------
    .with_column(
        Column("lr_gdp_pcap", from_loa="country_year", from_column="wdi_ny_gdp_pcap_kd")
            # .transform.missing.fill()
            # .transform.missing.replace_na()
            ) # GDP per capita

    #---------------------------------
    # Healthcare
    #---------------------------------
    # .with_column(
    #     Column("lr_ttns_zs", from_loa="country_year", from_column="wdi_sh_vac_ttns_zs")
    #         # .transform.missing.fill()
    #         # .transform.missing.replace_na()
    #         ) # Newborns protected against tetanus (%)

    #  .with_column(
    #     Column("lr_imrt_in", from_loa="country_year", from_column="wdi_sp_dyn_imrt_in")
    #         # .transform.missing.fill()
    #         # .transform.missing.replace_na()
    #         ) # Infant mortality rate
    
    # .with_column(
    #     Column("lr_le00_in", from_loa="country_year", from_column="wdi_sp_dyn_le00_in")
    #         # .transform.missing.fill()
    #         # .transform.missing.replace_na()
    #         ) # Life expectancy at birth

     .with_column(
        Column("lr_chex_gd_zs", from_loa="country_year", from_column="wdi_sh_xpd_chex_gd_zs")
            # .transform.missing.fill()
            # .transform.missing.replace_na()
            ) # Current health expenditure (% of GDP) further recompute:  multiply this by GDP per capita and divide by 100

    #---------------------------------
    # Food security
    #---------------------------------
    # .with_column(
    #     Column("lr_stnt_me_zs", from_loa="country_year", from_column="wdi_sh_sta_stnt_me_zs")
    #         # .transform.missing.fill()
    #         # .transform.missing.replace_na()
    #         ) # Prevalence of stunting/height for age (% of children under 5)

    .with_column(
        Column("lr_defc_zs", from_loa="country_year", from_column="wdi_sn_itk_defc_zs")
            # .transform.missing.fill()
            # .transform.missing.replace_na()
            )  # Prevalence of undernourisment (% of population)

    #---------------------------------
    # Education
    #---------------------------------
    # .with_column(
    #     Column("lr_prm_enrr", from_loa="country_year", from_column="wdi_se_prm_enrr")
    #         # .transform.missing.fill()
    #         # .transform.missing.replace_na()
    #         ) # School enrollment, primary

    #---------------------------------
    # Democracy
    #---------------------------------
    # .with_column(
    #     Column('lr_v2x_libdem', from_loa='country_year', from_column='vdem_v2x_libdem')
    #         # .transform.missing.fill()
    #         # .transform.missing.replace_na()
    #         )

    #---------------------------------
    # Population
    #---------------------------------
    .with_column(
        Column("lr_pop_totl", from_loa="country_year", from_column="wdi_sp_pop_totl")
            .transform.missing.fill()
            .transform.missing.replace_na()
            )
    
    )
    return queryset
