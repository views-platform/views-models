from viewser import Queryset, Column

def generate():
    """
    Contains the configuration for the input data in the form of a viewser queryset. That is the data from viewser that is used to train the model.
    This configuration is "behavioral" so modifying it will affect the model's runtime behavior and integration into the deployment system.
    There is no guarantee that the model will work if the input data configuration is changed here without changing the model settings and algorithm accordingly.

    Returns:
    - queryset_base (Queryset): A queryset containing the base data for the model training.
    """

    queryset_base = (Queryset("green_ranger", "country_month")

        .with_column(Column("lr_ns_best", from_loa="country_month", from_column="ged_ns_best_sum_nokgi")
            .transform.missing.replace_na())

        .with_column(Column("month", from_loa="month", from_column="month"))
        .with_column(Column("year_id", from_loa="country_year", from_column="year_id"))

    )

    return queryset_base
