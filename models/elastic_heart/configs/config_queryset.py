from viewser import Queryset, Column
from views_pipeline_core.managers.model import ModelPathManager

model_name = ModelPathManager.get_model_name_from_path(__file__)


def generate():
    """
    Contains the configuration for the input data in the form of a viewser queryset. That is the data from viewser that is used to train the model.
    This configuration is "behavioral" so modifying it will affect the model's runtime behavior and integration into the deployment system.
    There is no guarantee that the model will work if the input data configuration is changed here without changing the model settings and algorithm accordingly.

    Returns:
    - queryset_base (Queryset): A queryset containing the base data for the model training.
    """

    queryset = (
        Queryset(f"{model_name}", "country_month")
        .with_column(
            Column(
                "raw_ged_sb",
                from_loa="country_month",
                from_column="ged_sb_best_sum_nokgi",
            )
        )
        .with_column(
            Column(
                "raw_ged_ns",
                from_loa="country_month",
                from_column="ged_ns_best_sum_nokgi",
            )
        )
        .with_column(
            Column(
                "raw_ged_os",
                from_loa="country_month",
                from_column="ged_os_best_sum_nokgi",
            )
        )
        .with_column(
            Column("raw_acled_sb", from_loa="country_month", from_column="acled_sb_fat")
        )
        .with_column(
            Column(
                "raw_acled_sb_count",
                from_loa="country_month",
                from_column="acled_sb_count",
            )
        )
        .with_column(
            Column("raw_acled_os", from_loa="country_month", from_column="acled_os_fat")
        )
        .with_column(
            Column("raw_acled_ns", from_loa="country_month", from_column="acled_ns_fat")
        )
        .with_column(
            Column(
                "ln_ged_sb_dep",
                from_loa="country_month",
                from_column="ged_sb_best_sum_nokgi",
            )
            .transform.ops.ln()
            .transform.missing.fill()
            .transform.missing.replace_na()
        )
        .with_column(Column('lr_wdi_sp_pop_totl', from_loa='country_year', from_column='wdi_sp_pop_totl')
            .transform.missing.fill()
            .transform.missing.replace_na()
        )
        # .with_column(Column('ln_ged_ns_dep', from_loa='country_month', from_column='ged_ns_best_sum_nokgi')
        #     .transform.ops.ln()
        #     .transform.missing.fill()
        #     .transform.missing.replace_na()
        #     )
        # .with_column(Column('ln_ged_os_dep', from_loa='country_month', from_column='ged_os_best_sum_nokgi')
        #     .transform.ops.ln()
        #     .transform.missing.fill()
        #     .transform.missing.replace_na()
        # )
        .with_column(
            Column(
                "ln_ged_sb",
                from_loa="country_month",
                from_column="ged_sb_best_sum_nokgi",
            )
            .transform.ops.ln()
            .transform.missing.fill()
            .transform.missing.replace_na()
        )
        .with_column(
            Column(
                "ln_ged_ns",
                from_loa="country_month",
                from_column="ged_ns_best_sum_nokgi",
            )
            .transform.ops.ln()
            .transform.missing.fill()
            .transform.missing.replace_na()
        )
        .with_column(
            Column(
                "ln_ged_os",
                from_loa="country_month",
                from_column="ged_os_best_sum_nokgi",
            )
            .transform.ops.ln()
            .transform.missing.fill()
            .transform.missing.replace_na()
        )
        .with_column(
            Column("ln_acled_sb", from_loa="country_month", from_column="acled_sb_fat")
            .transform.ops.ln()
            .transform.missing.fill()
            .transform.missing.replace_na()
        )
        .with_column(
            Column(
                "ln_acled_sb_count",
                from_loa="country_month",
                from_column="acled_sb_count",
            )
            .transform.ops.ln()
            .transform.missing.fill()
            .transform.missing.replace_na()
        )
        # .with_column(
        #     Column("ln_acled_os", from_loa="country_month", from_column="acled_os_fat")
        #     .transform.ops.ln()
        #     .transform.missing.fill()
        #     .transform.missing.replace_na()
        # )
        # .with_column(
        #     Column(
        #         "lr_topic_tokens", from_loa="country_month", from_column="topic_tokens"
        #     )
        #     .transform.missing.fill()
        #     .transform.missing.replace_na()
        # )
        # .with_column(
        #     Column(
        #         "lr_topic_ste_theta4_stock",
        #         from_loa="country_month",
        #         from_column="topic_ste_theta4_stock",
        #     )
        #     .transform.missing.fill()
        #     .transform.missing.replace_na()
        # )
        # .with_column(
        #     Column(
        #         "lr_topic_ste_theta2_stock",
        #         from_loa="country_month",
        #         from_column="topic_ste_theta2_stock",
        #     )
        #     .transform.missing.fill()
        #     .transform.missing.replace_na()
        # )
        # # Unsure
        # .with_column(
        #     Column(
        #         "lr_topic_ste_theta0_stock",
        #         from_loa="country_month",
        #         from_column="topic_ste_theta0_stock",
        #     )
        #     .transform.missing.fill()
        #     .transform.missing.replace_na()
        # )
        # .with_column(
        #     Column(
        #         "lr_topic_ste_theta1_stock",
        #         from_loa="country_month",
        #         from_column="topic_ste_theta1_stock",
        #     )
        #     .transform.missing.fill()
        #     .transform.missing.replace_na()
        # )
        # .with_column(
        #     Column(
        #         "lr_topic_ste_theta3_stock",
        #         from_loa="country_month",
        #         from_column="topic_ste_theta3_stock",
        #     )
        #     .transform.missing.fill()
        #     .transform.missing.replace_na()
        # )
        # .with_column(
        #     Column(
        #         "lr_topic_ste_theta5_stock",
        #         from_loa="country_month",
        #         from_column="topic_ste_theta5_stock",
        #     )
        #     .transform.missing.fill()
        #     .transform.missing.replace_na()
        # )
        # .with_column(
        #     Column(
        #         "lr_topic_ste_theta6_stock",
        #         from_loa="country_month",
        #         from_column="topic_ste_theta6_stock",
        #     )
        #     .transform.missing.fill()
        #     .transform.missing.replace_na()
        # )
        # .with_column(
        #     Column(
        #         "lr_topic_ste_theta7_stock",
        #         from_loa="country_month",
        #         from_column="topic_ste_theta7_stock",
        #     )
        #     .transform.missing.fill()
        #     .transform.missing.replace_na()
        # )
        # .with_column(
        #     Column(
        #         "lr_topic_ste_theta8_stock",
        #         from_loa="country_month",
        #         from_column="topic_ste_theta8_stock",
        #     )
        #     .transform.missing.fill()
        #     .transform.missing.replace_na()
        # )
        # .with_column(
        #     Column(
        #         "lr_topic_ste_theta9_stock",
        #         from_loa="country_month",
        #         from_column="topic_ste_theta9_stock",
        #     )
        #     .transform.missing.fill()
        #     .transform.missing.replace_na()
        # )
        # .with_column(
        #     Column(
        #         "lr_topic_ste_theta10_stock",
        #         from_loa="country_month",
        #         from_column="topic_ste_theta10_stock",
        #     )
        #     .transform.missing.fill()
        #     .transform.missing.replace_na()
        # )
        # .with_column(
        #     Column(
        #         "lr_topic_ste_theta11_stock",
        #         from_loa="country_month",
        #         from_column="topic_ste_theta11_stock",
        #     )
        #     .transform.missing.fill()
        #     .transform.missing.replace_na()
        # )
        # .with_column(
        #     Column(
        #         "lr_topic_ste_theta12_stock",
        #         from_loa="country_month",
        #         from_column="topic_ste_theta12_stock",
        #     )
        #     .transform.missing.fill()
        #     .transform.missing.replace_na()
        # )
        # .with_column(
        #     Column(
        #         "lr_topic_ste_theta13_stock",
        #         from_loa="country_month",
        #         from_column="topic_ste_theta13_stock",
        #     )
        #     .transform.missing.fill()
        #     .transform.missing.replace_na()
        # )
        # .with_column(
        #     Column(
        #         "lr_topic_ste_theta14_stock",
        #         from_loa="country_month",
        #         from_column="topic_ste_theta14_stock",
        #     )
        #     .transform.missing.fill()
        #     .transform.missing.replace_na()
        # )
        .describe("""Base features for neural network models""")
    )

    return queryset
