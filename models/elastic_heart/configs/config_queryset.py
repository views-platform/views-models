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

    # VIEWSER 6, Example configuration. Modify as needed.

    def _add_conflict_history(queryset: Queryset) -> Queryset:
        print("Adding conflict history features...")
        return (
            queryset.with_column(
                Column(
                    "lr_ged_sb",
                    from_loa="country_month",
                    from_column="ged_sb_best_sum_nokgi",
                ).transform.missing.fill()
            )
            # .with_column(
            #     Column(
            #         "lr_ged_sb",
            #         from_loa="country_month",
            #         from_column="ged_sb_best_sum_nokgi",
            #     ).transform.missing.fill()
            # )
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
            .with_column(
                Column(
                    "lr_acled_sb", from_loa="country_month", from_column="acled_sb_fat"
                ).transform.missing.fill()
            )
            # REMOVED: lr_acled_sb_count - redundant with lr_acled_sb (same events, different measure)
            # .with_column(
            #     Column(
            #         "lr_acled_sb_count",
            #         from_loa="country_month",
            #         from_column="acled_sb_count",
            #     ).transform.missing.fill()
            # )
            .with_column(
                Column(
                    "lr_acled_os", from_loa="country_month", from_column="acled_os_fat"
                ).transform.missing.fill()
            )
            # .with_column(
            #     Column(
            #         "lr_ged_sb_tsum_24",
            #         from_loa="country_month",
            #         from_column="ged_sb_best_sum_nokgi",
            #     )
            #     .transform.missing.replace_na()
            #     .transform.temporal.moving_sum(24)
            #     .transform.missing.replace_na()
            # )
            .with_column(
                Column(
                    "lr_splag_1_ged_sb",
                    from_loa="country_month",
                    from_column="ged_sb_best_sum_nokgi",
                )
                .transform.missing.replace_na()
                .transform.spatial.countrylag(1, 1, 0, 0)
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_splag_1_ged_os",
                    from_loa="country_month",
                    from_column="ged_os_best_sum_nokgi",
                )
                .transform.missing.replace_na()
                .transform.spatial.countrylag(1, 1, 0, 0)
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_splag_1_ged_ns",
                    from_loa="country_month",
                    from_column="ged_ns_best_sum_nokgi",
                )
                .transform.missing.replace_na()
                .transform.spatial.countrylag(1, 1, 0, 0)
                .transform.missing.replace_na()
            )
        )

    def _add_wdi(queryset: Queryset) -> Queryset:
        return (
            queryset.with_column(
                Column(
                    "lr_wdi_sm_pop_netm",
                    from_loa="country_year",
                    from_column="wdi_sm_pop_netm",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_wdi_sm_pop_refg_or",
                    from_loa="country_year",
                    from_column="wdi_sm_pop_refg_or",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_wdi_dt_oda_odat_pc_zs",
                    from_loa="country_year",
                    from_column="wdi_dt_oda_odat_pc_zs",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_wdi_ms_mil_xpnd_gd_zs",
                    from_loa="country_year",
                    from_column="wdi_ms_mil_xpnd_gd_zs",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_wdi_sl_tlf_totl_fe_zs",
                    from_loa="country_year",
                    from_column="wdi_sl_tlf_totl_fe_zs",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_wdi_nv_agr_totl_kn",
                    from_loa="country_year",
                    from_column="wdi_nv_agr_totl_kn",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_wdi_sp_pop_grow",
                    from_loa="country_year",
                    from_column="wdi_sp_pop_grow",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_wdi_se_enr_prim_fm_zs",
                    from_loa="country_year",
                    from_column="wdi_se_enr_prim_fm_zs",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_wdi_sp_urb_totl_in_zs",
                    from_loa="country_year",
                    from_column="wdi_sp_urb_totl_in_zs",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_wdi_sh_sta_maln_zs",
                    from_loa="country_year",
                    from_column="wdi_sh_sta_maln_zs",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_wdi_sp_dyn_imrt_fe_in",
                    from_loa="country_year",
                    from_column="wdi_sp_dyn_imrt_fe_in",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_wdi_ny_gdp_mktp_kd",
                    from_loa="country_year",
                    from_column="wdi_ny_gdp_mktp_kd",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_wdi_sh_sta_stnt_zs",
                    from_loa="country_year",
                    from_column="wdi_sh_sta_stnt_zs",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
        )

    def _add_vdem(queryset: Queryset) -> Queryset:
        return (
            queryset.with_column(
                Column(
                    "lr_vdem_v2x_horacc",
                    from_loa="country_year",
                    from_column="vdem_v2x_horacc",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_vdem_v2xnp_client",
                    from_loa="country_year",
                    from_column="vdem_v2xnp_client",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_vdem_v2x_veracc",
                    from_loa="country_year",
                    from_column="vdem_v2x_veracc",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_vdem_v2x_divparctrl",
                    from_loa="country_year",
                    from_column="vdem_v2x_divparctrl",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_vdem_v2xpe_exlpol",
                    from_loa="country_year",
                    from_column="vdem_v2xpe_exlpol",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_vdem_v2x_diagacc",
                    from_loa="country_year",
                    from_column="vdem_v2x_diagacc",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_vdem_v2xpe_exlgeo",
                    from_loa="country_year",
                    from_column="vdem_v2xpe_exlgeo",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_vdem_v2xpe_exlgender",
                    from_loa="country_year",
                    from_column="vdem_v2xpe_exlgender",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_vdem_v2xpe_exlsocgr",
                    from_loa="country_year",
                    from_column="vdem_v2xpe_exlsocgr",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_vdem_v2x_ex_party",
                    from_loa="country_year",
                    from_column="vdem_v2x_ex_party",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_vdem_v2x_genpp",
                    from_loa="country_year",
                    from_column="vdem_v2x_genpp",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_vdem_v2xeg_eqdr",
                    from_loa="country_year",
                    from_column="vdem_v2xeg_eqdr",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_vdem_v2xcl_prpty",
                    from_loa="country_year",
                    from_column="vdem_v2xcl_prpty",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_vdem_v2xeg_eqprotec",
                    from_loa="country_year",
                    from_column="vdem_v2xeg_eqprotec",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_vdem_v2x_ex_military",
                    from_loa="country_year",
                    from_column="vdem_v2x_ex_military",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_vdem_v2xcl_dmove",
                    from_loa="country_year",
                    from_column="vdem_v2xcl_dmove",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_vdem_v2x_clphy",
                    from_loa="country_year",
                    from_column="vdem_v2x_clphy",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            # REMOVED: lr_vdem_v2x_hosabort - 99.8% zeros, effectively constant (no predictive value)
            # .with_column(
            #     Column(
            #         "lr_vdem_v2x_hosabort",
            #         from_loa="country_year",
            #         from_column="vdem_v2x_hosabort",
            #     )
            #     .transform.missing.fill()
            #     .transform.missing.replace_na()
            # )
            .with_column(
                Column(
                    "lr_vdem_v2xnp_regcorr",
                    from_loa="country_year",
                    from_column="vdem_v2xnp_regcorr",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
        )

    def _add_topics(queryset: Queryset) -> Queryset:
        print("Adding topic model features...")
        # REMOVED: lr_topic_tokens_t1 - extreme skew (81.6), ~40% zeros, redundant with theta features
        # REMOVED: lr_topic_tokens_t1_splag - extreme max (~858K), scaling issues
        # The theta probability distributions capture topic information more stably
        return (
            queryset
            # .with_column(
            #     Column(
            #         "lr_topic_tokens_t1",
            #         from_loa="country_month",
            #         from_column="topic_tokens",
            #     )
            #     .transform.missing.fill()
            #     .transform.missing.replace_na()
            # )
            .with_column(
                Column(
                    "lr_topic_ste_theta0",
                    from_loa="country_month",
                    from_column="topic_ste_theta0_stock",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_topic_ste_theta1",
                    from_loa="country_month",
                    from_column="topic_ste_theta1_stock",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_topic_ste_theta2",
                    from_loa="country_month",
                    from_column="topic_ste_theta2_stock",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_topic_ste_theta3",
                    from_loa="country_month",
                    from_column="topic_ste_theta3_stock",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_topic_ste_theta4",
                    from_loa="country_month",
                    from_column="topic_ste_theta4_stock",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_topic_ste_theta5",
                    from_loa="country_month",
                    from_column="topic_ste_theta5_stock",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_topic_ste_theta6",
                    from_loa="country_month",
                    from_column="topic_ste_theta6_stock",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_topic_ste_theta7",
                    from_loa="country_month",
                    from_column="topic_ste_theta7_stock",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_topic_ste_theta8",
                    from_loa="country_month",
                    from_column="topic_ste_theta8_stock",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_topic_ste_theta9",
                    from_loa="country_month",
                    from_column="topic_ste_theta9_stock",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_topic_ste_theta10",
                    from_loa="country_month",
                    from_column="topic_ste_theta10_stock",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_topic_ste_theta11",
                    from_loa="country_month",
                    from_column="topic_ste_theta11_stock",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_topic_ste_theta12",
                    from_loa="country_month",
                    from_column="topic_ste_theta12_stock",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_topic_ste_theta13",
                    from_loa="country_month",
                    from_column="topic_ste_theta13_stock",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_topic_ste_theta14",
                    from_loa="country_month",
                    from_column="topic_ste_theta14_stock",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            # REMOVED: lr_topic_tokens_t1_splag - extreme max (~858K), scaling issues
            # .with_column(
            #     Column(
            #         "lr_topic_tokens_t1_splag",
            #         from_loa="country_month",
            #         from_column="topic_tokens",
            #     )
            #     .transform.missing.fill()
            #     .transform.missing.replace_na()
            #     .transform.spatial.countrylag(1, 1, 0, 0)
            #     .transform.missing.replace_na()
            # )
            .with_column(
                Column(
                    "lr_topic_ste_theta0_stock_t1_splag",
                    from_loa="country_month",
                    from_column="topic_ste_theta0_stock",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
                .transform.spatial.countrylag(1, 1, 0, 0)
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_topic_ste_theta1_stock_t1_splag",
                    from_loa="country_month",
                    from_column="topic_ste_theta1_stock",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
                .transform.spatial.countrylag(1, 1, 0, 0)
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_topic_ste_theta2_stock_t1_splag",
                    from_loa="country_month",
                    from_column="topic_ste_theta2_stock",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
                .transform.spatial.countrylag(1, 1, 0, 0)
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_topic_ste_theta3_stock_t1_splag",
                    from_loa="country_month",
                    from_column="topic_ste_theta3_stock",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
                .transform.spatial.countrylag(1, 1, 0, 0)
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_topic_ste_theta4_stock_t1_splag",
                    from_loa="country_month",
                    from_column="topic_ste_theta4_stock",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
                .transform.spatial.countrylag(1, 1, 0, 0)
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_topic_ste_theta5_stock_t1_splag",
                    from_loa="country_month",
                    from_column="topic_ste_theta5_stock",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
                .transform.spatial.countrylag(1, 1, 0, 0)
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_topic_ste_theta6_stock_t1_splag",
                    from_loa="country_month",
                    from_column="topic_ste_theta6_stock",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
                .transform.spatial.countrylag(1, 1, 0, 0)
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_topic_ste_theta7_stock_t1_splag",
                    from_loa="country_month",
                    from_column="topic_ste_theta7_stock",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
                .transform.spatial.countrylag(1, 1, 0, 0)
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_topic_ste_theta8_stock_t1_splag",
                    from_loa="country_month",
                    from_column="topic_ste_theta8_stock",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
                .transform.spatial.countrylag(1, 1, 0, 0)
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_topic_ste_theta9_stock_t1_splag",
                    from_loa="country_month",
                    from_column="topic_ste_theta9_stock",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
                .transform.spatial.countrylag(1, 1, 0, 0)
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_topic_ste_theta10_stock_t1_splag",
                    from_loa="country_month",
                    from_column="topic_ste_theta10_stock",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
                .transform.spatial.countrylag(1, 1, 0, 0)
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_topic_ste_theta11_stock_t1_splag",
                    from_loa="country_month",
                    from_column="topic_ste_theta11_stock",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
                .transform.spatial.countrylag(1, 1, 0, 0)
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_topic_ste_theta12_stock_t1_splag",
                    from_loa="country_month",
                    from_column="topic_ste_theta12_stock",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
                .transform.spatial.countrylag(1, 1, 0, 0)
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_topic_ste_theta13_stock_t1_splag",
                    from_loa="country_month",
                    from_column="topic_ste_theta13_stock",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
                .transform.spatial.countrylag(1, 1, 0, 0)
                .transform.missing.replace_na()
            )
            .with_column(
                Column(
                    "lr_topic_ste_theta14_stock_t1_splag",
                    from_loa="country_month",
                    from_column="topic_ste_theta14_stock",
                )
                .transform.missing.fill()
                .transform.missing.replace_na()
                .transform.spatial.countrylag(1, 1, 0, 0)
                .transform.missing.replace_na()
            )
        )

    queryset = Queryset(f"{model_name}", "country_month")

    return _add_topics(_add_vdem(_add_wdi(_add_conflict_history(queryset))))
    # return _add_topics(_add_conflict_history(queryset=queryset))
