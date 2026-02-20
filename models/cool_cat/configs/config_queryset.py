from viewser import Queryset, Column
from views_pipeline_core.managers.model import ModelPathManager

model_name = ModelPathManager.get_model_name_from_path(__file__)


def generate():
    """
    Optimized queryset for conflict forecasting with TiDE + NegativeBinomialLoss.
    
    Feature Strategy:
    -----------------
    1. Conflict History (core signal): GED counts + momentum (deltas)
    2. Spatial Spillover: Neighboring country conflict only
    3. Seasonality: Raw month (1-12), scaled via MinMaxScaler
    4. Governance (V-Dem): Full set - strong predictors of institutional fragility
    5. Economic Stress (WDI): Full set - economic shocks trigger unrest
    6. Topic Models: Top 7 topics (reduced from 15)

    """

    def _add_conflict_history(queryset: Queryset) -> Queryset:
        """Core conflict features: counts, momentum, spatial spillover."""
        return (
            queryset
            # ==================== TARGETS ====================
            .with_column(
                Column("lr_ged_sb", from_loa="country_month", from_column="ged_sb_best_sum_nokgi")
                .transform.missing.fill()
            )
            .with_column(
                Column("lr_ged_ns", from_loa="country_month", from_column="ged_ns_best_sum_nokgi")
                .transform.missing.fill()
            )
            .with_column(
                Column("lr_ged_os", from_loa="country_month", from_column="ged_os_best_sum_nokgi")
                .transform.missing.fill()
            )
            # ==================== MOMENTUM (DELTAS) ====================
            .with_column(
                Column("lr_ged_sb_delta", from_loa="country_month", from_column="ged_sb_best_sum_nokgi")
                .transform.missing.fill()
                .transform.temporal.delta(1)
                .transform.missing.replace_na()
            )
            .with_column(
                Column("lr_ged_ns_delta", from_loa="country_month", from_column="ged_ns_best_sum_nokgi")
                .transform.missing.fill()
                .transform.temporal.delta(1)
                .transform.missing.replace_na()
            )
            .with_column(
                Column("lr_ged_os_delta", from_loa="country_month", from_column="ged_os_best_sum_nokgi")
                .transform.missing.fill()
                .transform.temporal.delta(1)
                .transform.missing.replace_na()
            )
            # ==================== SPATIAL SPILLOVER ====================
            .with_column(
                Column("lr_splag_1_ged_sb", from_loa="country_month", from_column="ged_sb_best_sum_nokgi")
                .transform.missing.replace_na()
                .transform.spatial.countrylag(1, 1, 0, 0)
                .transform.missing.replace_na()
            )
            .with_column(
                Column("lr_splag_1_ged_ns", from_loa="country_month", from_column="ged_ns_best_sum_nokgi")
                .transform.missing.replace_na()
                .transform.spatial.countrylag(1, 1, 0, 0)
                .transform.missing.replace_na()
            )
            .with_column(
                Column("lr_splag_1_ged_os", from_loa="country_month", from_column="ged_os_best_sum_nokgi")
                .transform.missing.replace_na()
                .transform.spatial.countrylag(1, 1, 0, 0)
                .transform.missing.replace_na()
            )
            # ==================== SEASONALITY ====================
            .with_column(
                Column("month", from_loa="country_month", from_column="month")
                .transform.missing.fill()
            )
        )

    def _add_wdi(queryset: Queryset) -> Queryset:
        """Economic indicators from World Development Indicators."""
        return (
            queryset
            # Migration & Refugees
            .with_column(
                Column("lr_wdi_sm_pop_netm", from_loa="country_year", from_column="wdi_sm_pop_netm")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_wdi_sm_pop_refg_or", from_loa="country_year", from_column="wdi_sm_pop_refg_or")
                .transform.missing.fill().transform.missing.replace_na()
            )
            # Aid & Military
            .with_column(
                Column("lr_wdi_dt_oda_odat_pc_zs", from_loa="country_year", from_column="wdi_dt_oda_odat_pc_zs")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_wdi_ms_mil_xpnd_gd_zs", from_loa="country_year", from_column="wdi_ms_mil_xpnd_gd_zs")
                .transform.missing.fill().transform.missing.replace_na()
            )
            # Labor & Demographics
            .with_column(
                Column("lr_wdi_sl_tlf_totl_fe_zs", from_loa="country_year", from_column="wdi_sl_tlf_totl_fe_zs")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_wdi_sp_pop_grow", from_loa="country_year", from_column="wdi_sp_pop_grow")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_wdi_sp_urb_totl_in_zs", from_loa="country_year", from_column="wdi_sp_urb_totl_in_zs")
                .transform.missing.fill().transform.missing.replace_na()
            )
            # Education & Health
            .with_column(
                Column("lr_wdi_se_enr_prim_fm_zs", from_loa="country_year", from_column="wdi_se_enr_prim_fm_zs")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_wdi_sp_dyn_imrt_fe_in", from_loa="country_year", from_column="wdi_sp_dyn_imrt_fe_in")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_wdi_sh_sta_maln_zs", from_loa="country_year", from_column="wdi_sh_sta_maln_zs")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_wdi_sh_sta_stnt_zs", from_loa="country_year", from_column="wdi_sh_sta_stnt_zs")
                .transform.missing.fill().transform.missing.replace_na()
            )
            # Economy
            .with_column(
                Column("lr_wdi_ny_gdp_mktp_kd", from_loa="country_year", from_column="wdi_ny_gdp_mktp_kd")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_wdi_nv_agr_totl_kn", from_loa="country_year", from_column="wdi_nv_agr_totl_kn")
                .transform.missing.fill().transform.missing.replace_na()
            )
        )

    def _add_vdem(queryset: Queryset) -> Queryset:
        """Governance indicators from V-Dem."""
        return (
            queryset
            # Accountability & Transparency
            .with_column(
                Column("lr_vdem_v2x_horacc", from_loa="country_year", from_column="vdem_v2x_horacc")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_vdem_v2x_veracc", from_loa="country_year", from_column="vdem_v2x_veracc")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_vdem_v2x_diagacc", from_loa="country_year", from_column="vdem_v2x_diagacc")
                .transform.missing.fill().transform.missing.replace_na()
            )
            # Clientelism & Corruption
            .with_column(
                Column("lr_vdem_v2xnp_client", from_loa="country_year", from_column="vdem_v2xnp_client")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_vdem_v2xnp_regcorr", from_loa="country_year", from_column="vdem_v2xnp_regcorr")
                .transform.missing.fill().transform.missing.replace_na()
            )
            # Political Exclusion
            .with_column(
                Column("lr_vdem_v2xpe_exlpol", from_loa="country_year", from_column="vdem_v2xpe_exlpol")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_vdem_v2xpe_exlgeo", from_loa="country_year", from_column="vdem_v2xpe_exlgeo")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_vdem_v2xpe_exlgender", from_loa="country_year", from_column="vdem_v2xpe_exlgender")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_vdem_v2xpe_exlsocgr", from_loa="country_year", from_column="vdem_v2xpe_exlsocgr")
                .transform.missing.fill().transform.missing.replace_na()
            )
            # Party & Executive
            .with_column(
                Column("lr_vdem_v2x_divparctrl", from_loa="country_year", from_column="vdem_v2x_divparctrl")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_vdem_v2x_ex_party", from_loa="country_year", from_column="vdem_v2x_ex_party")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_vdem_v2x_ex_military", from_loa="country_year", from_column="vdem_v2x_ex_military")
                .transform.missing.fill().transform.missing.replace_na()
            )
            # Civil Liberties & Rights
            .with_column(
                Column("lr_vdem_v2x_genpp", from_loa="country_year", from_column="vdem_v2x_genpp")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_vdem_v2xeg_eqdr", from_loa="country_year", from_column="vdem_v2xeg_eqdr")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_vdem_v2xcl_prpty", from_loa="country_year", from_column="vdem_v2xcl_prpty")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_vdem_v2xeg_eqprotec", from_loa="country_year", from_column="vdem_v2xeg_eqprotec")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_vdem_v2xcl_dmove", from_loa="country_year", from_column="vdem_v2xcl_dmove")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_vdem_v2x_clphy", from_loa="country_year", from_column="vdem_v2x_clphy")
                .transform.missing.fill().transform.missing.replace_na()
            )
        )

    def _add_topics(queryset: Queryset) -> Queryset:
        """Top 7 topic model features. Removed topics 7-14 and all spatial lags."""
        return (
            queryset
            .with_column(
                Column("lr_topic_ste_theta0", from_loa="country_month", from_column="topic_ste_theta0_stock")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_topic_ste_theta1", from_loa="country_month", from_column="topic_ste_theta1_stock")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_topic_ste_theta2", from_loa="country_month", from_column="topic_ste_theta2_stock")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_topic_ste_theta3", from_loa="country_month", from_column="topic_ste_theta3_stock")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_topic_ste_theta4", from_loa="country_month", from_column="topic_ste_theta4_stock")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_topic_ste_theta5", from_loa="country_month", from_column="topic_ste_theta5_stock")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_topic_ste_theta6", from_loa="country_month", from_column="topic_ste_theta6_stock")
                .transform.missing.fill().transform.missing.replace_na()
            )
        )

    queryset = Queryset(f"{model_name}", "country_month")
    return _add_topics(_add_vdem(_add_wdi(_add_conflict_history(queryset))))
