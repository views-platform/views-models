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

    queryset_base = (Queryset("uncertainty_hh_baseline", "country_month")

        # target variable
        .with_column(Column("ged_sb_dep", from_loa="country_month", from_column="ged_sb_best_sum_nokgi")
                        .transform.missing.fill()
                        )

        # timelag 0-2 of target variable
        .with_column(Column("ged_sb", from_loa="country_month", from_column="ged_sb_best_sum_nokgi")
                        .transform.missing.fill()
                        )
        
        .with_column(Column("ged_sb_tlag_1", from_loa="country_month",
                        from_column="ged_sb_best_sum_nokgi")
                .transform.temporal.tlag(1)
                .transform.missing.fill()
        .transform.missing.replace_na()
                )

        .with_column(Column("ged_sb_tlag_2", from_loa="country_month",
                        from_column="ged_sb_best_sum_nokgi")
                .transform.temporal.tlag(2)
                .transform.missing.fill()
        .transform.missing.replace_na()
                )
        # Decay functions
        # sb
        .with_column(Column("decay_ged_sb_5", from_loa="country_month", from_column="ged_sb_best_sum_nokgi")
                        .transform.missing.replace_na()
                        .transform.bool.gte(5)
                        .transform.temporal.time_since()
                        .transform.temporal.decay(24)
                        .transform.missing.replace_na()
                        )
        
        .with_column(Column("decay_ged_sb_100", from_loa="country_month",
                        from_column="ged_sb_best_sum_nokgi")
                .transform.missing.replace_na()
                .transform.bool.gte(500)
                .transform.temporal.time_since()
                .transform.temporal.decay(24)
                .transform.missing.replace_na()
                )
        # os
        .with_column(Column("decay_ged_os_5", from_loa="country_month", from_column="ged_os_best_sum_nokgi")
                        .transform.missing.replace_na()
                        .transform.bool.gte(5)
                        .transform.temporal.time_since()
                        .transform.temporal.decay(24)
                        .transform.missing.replace_na()
                        )
        # Spatial lag decay
        .with_column(Column("splag_1_decay_ged_sb_5", from_loa="country_month",
                                from_column="ged_sb_best_sum_nokgi")
                        .transform.missing.replace_na()
                        .transform.bool.gte(5)
                        .transform.temporal.time_since()
                        .transform.temporal.decay(24)
                        .transform.spatial.countrylag(1, 1, 0, 0)
                        .transform.missing.replace_na()
                        )
# ns: ['', 'vdem_v12_v2xpe_exlpol', 'vdem_v12_v2xpe_exlgeo', 'vdem_v12_v2xcl_rol', 'vdem_v12_v2x_libdem']" 

        # From WDI
        .with_column(Column("wdi_sp_pop_totl", from_loa="country_year", from_column="wdi_sp_pop_totl")
                        .transform.missing.fill()
                        .transform.temporal.tlag(12)
                        .transform.missing.fill()
                        .transform.missing.replace_na()
                        )
        .with_column(Column("wdi_gdp_pcap_kd", from_loa="country_year", from_column="wdi_ny_gdp_pcap_kd")
                        .transform.missing.fill()
                        .transform.temporal.tlag(12)
                        .transform.missing.fill()
                        .transform.missing.replace_na()
                        )
        
        .with_column(Column("wdi_sp_dyn_imrt_fe_in", from_loa="country_year",
                        from_column="wdi_sp_dyn_imrt_fe_in")
                        .transform.missing.fill()
                        .transform.temporal.tlag(12)
                        .transform.missing.fill()
                        .transform.missing.replace_na()
                        )
        .with_column(Column("splag_wdi_gdp_pcap_kd", from_loa="country_year",
                        from_column="wdi_ny_gdp_pcap_kd")
                        .transform.missing.fill()
                        .transform.temporal.tlag(12)
                        .transform.spatial.countrylag(1, 1, 0, 0)
                        .transform.missing.replace_na()
                        )
        
        # From vdem
        



        .with_theme("fatalities")
        .describe("""Fatalities conflict history, cm level

        Predicting fatalities using conflict predictors, ultrashort

                        """)
        )

    return queryset_base

if __name__ == "__main__":
    generate()