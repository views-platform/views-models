from viewser import Queryset, Column

def generate():
    queryset = (Queryset("wdi_tempdisagg_nochange","country_month")

    .with_column(
        Column("lr_gdp_pcap", from_loa="country_year", from_column="wdi_ny_gdp_pcap_kd")
            .transform.missing.fill()
            .transform.missing.replace_na()
            )
    
    )
    return queryset
