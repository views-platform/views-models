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

    queryset = (Queryset('structural_brief_nolog','country_month')
        .with_column(Column('lr_gleditsch_ward', from_loa='country', from_column='gwcode')
            )

        .with_column(Column('lr_sb_best', from_loa='country_month', from_column='ged_sb_best_sum_nokgi')
            .transform.missing.fill()
            .transform.missing.replace_na()
            )


        .with_column(Column('lr_wdi_ny_gdp_mktp_kd', from_loa='country_year', from_column='wdi_ny_gdp_mktp_kd')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )


        .with_column(Column('lr_vdem_v2x_libdem', from_loa='country_year', from_column='vdem_v2x_libdem')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )


        .with_column(Column('lr_wdi_sp_pop_totl', from_loa='country_year', from_column='wdi_sp_pop_totl')
            .transform.missing.fill()
            .transform.temporal.tlag(12)
            .transform.missing.fill()
            .transform.missing.replace_na()
            )



        .with_theme('uncertainty')
        .describe("""Predicting fatalities, cm level
        
                                Queryset with a small number of structural features, no conflict history
        
                                """)
        )

    return queryset