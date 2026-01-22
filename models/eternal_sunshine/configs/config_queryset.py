from viewser import Queryset, Column
from views_pipeline_core.managers.model import ModelPathManager

model_name = ModelPathManager.get_model_name_from_path(__file__)


def generate():

    qs_broad = (
        Queryset(f"{model_name}", "priogrid_month")
        .with_column(
            Column(
                "lr_ged_os",
                from_loa="priogrid_month",
                from_column="ged_os_best_sum_nokgi",
            )
            .transform.missing.fill()
            .transform.missing.replace_na()
        )
        .with_column(
            Column(
                "lr_ged_ns",
                from_loa="priogrid_month",
                from_column="ged_ns_best_sum_nokgi",
            )
            .transform.missing.fill()
            .transform.missing.replace_na()
        )
        .with_column(
            Column(
                "lr_tlag1_dr_mod_gs",
                from_loa="priogrid_month",
                from_column="tlag1_dr_mod_gs",
            ).transform.missing.replace_na(0)
        )
        .with_column(
            Column(
                "lr_spei1_gs_prev10_anom",
                from_loa="priogrid_month",
                from_column="spei1_gs_prev10_anom",
            ).transform.missing.replace_na(0)
        )
        .with_column(
            Column(
                "lr_spei1gsy_lowermedian_count",
                from_loa="priogrid_month",
                from_column="spei1gsy_lowermedian_count",
            ).transform.missing.replace_na(0)
        )
        .with_column(
            Column(
                "lr_ged_sb",
                from_loa="priogrid_month",
                from_column="ged_sb_best_sum_nokgi",
            )
            .transform.missing.fill()
            .transform.missing.replace_na()
        )
        .with_column(
            Column(
                "lr_treelag_1_sb",
                from_loa="priogrid_month",
                from_column="ged_sb_best_sum_nokgi",
            )
            .transform.missing.replace_na()
            .transform.spatial.treelag(0.7, 1)
        )
        .with_column(
            Column(
                "lr_treelag_2_sb",
                from_loa="priogrid_month",
                from_column="ged_sb_best_sum_nokgi",
            )
            .transform.missing.replace_na()
            .transform.spatial.treelag(0.7, 2)
        )
        .with_column(
            Column(
                "lr_sptime_dist_k1_ged_sb",
                from_loa="priogrid_month",
                from_column="ged_sb_best_sum_nokgi",
            )
            .transform.missing.replace_na()
            .transform.spatial.sptime_dist("distances", 1, 1.0, 0.0)
        )
        .with_column(
            Column(
                "lr_sptime_dist_k10_ged_sb",
                from_loa="priogrid_month",
                from_column="ged_sb_best_sum_nokgi",
            )
            .transform.missing.replace_na()
            .transform.spatial.sptime_dist("distances", 1, 10.0, 0.0)
        )
        .with_column(
            Column(
                "lr_sptime_dist_k001_ged_sb",
                from_loa="priogrid_month",
                from_column="ged_sb_best_sum_nokgi",
            )
            .transform.missing.replace_na()
            .transform.spatial.sptime_dist("distances", 1, 0.01, 0.0)
        )
        .with_column(
            Column(
                "lr_dist_diamsec", from_loa="priogrid", from_column="dist_diamsec_s_wgs"
            )
            .transform.missing.fill()
            .transform.missing.replace_na()
        )
        .with_column(
            Column("lr_imr_mean", from_loa="priogrid_year", from_column="imr_mean")
            .transform.missing.fill()
            .transform.missing.replace_na()
        )
        .with_column(
            Column("ln_ttime_mean", from_loa="priogrid_year", from_column="ttime_mean")
            .transform.ops.ln()
            .transform.missing.fill()
            .transform.missing.replace_na()
        )
        .with_column(
            Column("ln_bdist3", from_loa="priogrid_year", from_column="bdist3")
            .transform.ops.ln()
            .transform.missing.fill()
            .transform.missing.replace_na()
        )
        .with_column(
            Column("ln_capdist", from_loa="priogrid_year", from_column="capdist")
            .transform.ops.ln()
            .transform.missing.fill()
            .transform.missing.replace_na()
        )
        .with_column(
            Column(
                "ln_pop_gpw_sum", from_loa="priogrid_year", from_column="pop_gpw_sum"
            )
            .transform.ops.ln()
            .transform.missing.fill()
            .transform.missing.replace_na()
        )
        .with_column(
            Column(
                "lr_decay_ged_sb_5",
                from_loa="priogrid_month",
                from_column="ged_sb_best_sum_nokgi",
            )
            .transform.missing.replace_na()
            .transform.bool.gte(5)
            .transform.temporal.time_since()
            .transform.temporal.decay(12)
            .transform.missing.replace_na()
        )
        .with_column(
            Column(
                "lr_decay_ged_os_5",
                from_loa="priogrid_month",
                from_column="ged_os_best_sum_nokgi",
            )
            .transform.missing.replace_na()
            .transform.bool.gte(5)
            .transform.temporal.time_since()
            .transform.temporal.decay(12)
            .transform.missing.replace_na()
        )
        .with_column(
            Column(
                "lr_decay_ged_ns_5",
                from_loa="priogrid_month",
                from_column="ged_ns_best_sum_nokgi",
            )
            .transform.missing.replace_na()
            .transform.bool.gte(5)
            .transform.temporal.time_since()
            .transform.temporal.decay(12)
            .transform.missing.replace_na()
        )
        .with_column(
            Column(
                "lr_splag_1_1_sb_1",
                from_loa="priogrid_month",
                from_column="ged_sb_best_sum_nokgi",
            )
            .transform.missing.replace_na()
            .transform.bool.gte(1)
            .transform.temporal.time_since()
            .transform.temporal.decay(24)
            .transform.spatial.lag(1, 1, 0, 0)
            .transform.missing.replace_na()
        )
    )

    return qs_broad
