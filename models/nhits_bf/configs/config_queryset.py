from viewser import Queryset, Column

def generate():
    queryset = (Queryset("nhits_bf","country_month")

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
        Column('lr_vdem_v2x_libdem', from_loa='country_year', from_column='vdem_v2x_libdem')
            .transform.missing.fill()
            .transform.missing.replace_na()
            )
    )
    return queryset
