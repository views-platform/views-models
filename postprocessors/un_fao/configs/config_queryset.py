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

    queryset_base = (Queryset("un_fao", "priogrid_month")
        .with_column(Column("lr_ged_sb", from_loa="priogrid_month", from_column="ged_sb_best_sum_nokgi")
            .transform.missing.replace_na())
        .with_column(Column("lr_ged_ns", from_loa="priogrid_month", from_column="ged_ns_best_sum_nokgi")
            .transform.missing.replace_na())
        .with_column(Column("lr_ged_os", from_loa="priogrid_month", from_column="ged_os_best_sum_nokgi")
            .transform.missing.replace_na())
    )

    return queryset_base
