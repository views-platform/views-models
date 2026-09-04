from viewser import Queryset, Column
from views_pipeline_core.managers.model import ModelPathManager

model_name = ModelPathManager.get_model_name_from_path(__file__)


def generate():
    """
    Country-month conflict-history-only queryset.
    """
    queryset = (
        Queryset(f"{model_name}", "country_month")
        .with_column(
            Column(
                "lr_ged_sb",
                from_loa="country_month",
                from_column="ged_sb_best_sum_nokgi",
            ).transform.missing.fill()
        )
        .with_column(
            Column(
                "lr_ged_ns",
                from_loa="country_month",
                from_column="ged_ns_best_sum_nokgi",
            ).transform.missing.fill()
        )
        .with_column(
            Column(
                "lr_ged_os",
                from_loa="country_month",
                from_column="ged_os_best_sum_nokgi",
            ).transform.missing.fill()
        )
    )

    return queryset
