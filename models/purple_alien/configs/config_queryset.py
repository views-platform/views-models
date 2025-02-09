from viewser import Queryset, Column

def generate():
    """
    Contains the configuration for the input data in the form of a viewser queryset. That is the data from viewser that is used to train the model.
    This configuration is "behavioral" so modifying it will affect the model's runtime behavior and integration into the deployment system.
    There is no guarantee that the model will work if the input data configuration is changed here without changing the model settings and algorithm accordingly.

    Returns:
    - queryset_base (Queryset): A queryset containing the base data for the model training.
    """
    
    # VIEWSER 6

    queryset_base = (Queryset("purple_alien", "priogrid_month")
        .with_column(Column("ln_sb_best", from_loa = "priogrid_month", from_column = "ged_sb_best_sum_nokgi").transform.ops.ln().transform.missing.replace_na())
        .with_column(Column("ln_ns_best", from_loa = "priogrid_month", from_column = "ged_ns_best_sum_nokgi").transform.ops.ln().transform.missing.replace_na())
        .with_column(Column("ln_os_best", from_loa = "priogrid_month", from_column = "ged_os_best_sum_nokgi").transform.ops.ln().transform.missing.replace_na())
        .with_column(Column("month", from_loa = "month", from_column = "month"))
        .with_column(Column("year_id", from_loa = "country_year", from_column = "year_id"))
        .with_column(Column("c_id", from_loa = "country_year", from_column = "country_id"))
        .with_column(Column("col", from_loa = "priogrid", from_column = "col"))
        .with_column(Column("row", from_loa = "priogrid", from_column = "row")))


    return queryset_base
