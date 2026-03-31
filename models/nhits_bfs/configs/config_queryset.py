from viewser import Queryset, Column

def generate():
    queryset = (Queryset("nhits_bfs","country_month")

    .with_column(
        Column("lr_gdp_pcap", from_loa="country_year", from_column="wdi_ny_gdp_pcap_kd")
            .transform.missing.fill()
            .transform.missing.replace_na()
            )


    .with_column(
        Column("lr_pop_totl", from_loa="country_year", from_column="wdi_sp_pop_totl")
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

    .with_column(
        Column("lr_pop_totl_splag", from_loa="country_year", from_column="wdi_sp_pop_totl")
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.spatial.countrylag(1, 1, 0, 0)
            )
            
    .with_column(Column('lr_vdem_v2x_libdem', from_loa='country_year', from_column='vdem_v2x_libdem')
            .transform.missing.fill()
            .transform.missing.replace_na()
            )
            
    .with_column(
        Column("lr_vdem_v2x_libdem_splag", from_loa="country_year", from_column="vdem_v2x_libdem")
            .transform.missing.fill()
            .transform.missing.replace_na()
            .transform.spatial.countrylag(1, 1, 0, 0)
            )
    )
    return queryset
