from viewser import Queryset, Column
from views_pipeline_core.managers.model import ModelPathManager

model_name = ModelPathManager.get_model_name_from_path(__file__)


def generate():
    """
    Queryset for conflict forecasting with TiDE + SpotlightLossLogcosh.

    Feature Strategy:
    -----------------
    1. Conflict History:   GED counts + deltas + ACLED counts/fatalities + spatial lags
    2. Conflict Memory:    Exponential decay features (time since threshold crossed) —
                           the primary fix for the annual-feature smoothing problem.
                           Decay features are country-month level, update every step,
                           and encode "is this country currently in a conflict regime?"
                           without level-leakage (bounded ∈ [0,1]).
    3. Temporal Lags:      ln_ged explicit lags (t-1..t-6) — TiDE has no recurrence,
                           so explicit temporal context must be provided.
    4. Topic/NLP:          News topic model stocks (t-1, t-2, t-13 + spatial lags) —
                           leading indicators not present in conflict counts.
    5. Governance (V-Dem): Pruned set — removed redundant accountability/exclusion indices.
    6. Economic Stress (WDI): Pruned set — removed redundant development proxies.

    Feature Counts (with USE_STATIC_COVS=True):
    - Conflict counts + delta + splag: 12
    - Decay + splag decay:             13
    - ln_ged temporal lags:             7
    - Topic:                           10
    - WDI:                              8
    - V-Dem:                           12
    - Total:                           ~62 features
    """

    USE_STATIC_COVS = True  # Toggle based on model support

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
            # ACLED event COUNT only — fatalities redundant with GED
            .with_column(Column('lr_acled_sb', from_loa='country_month', from_column='acled_sb_fat')
            # .transform.ops.ln()
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

            .with_column(Column('lr_acled_sb_count', from_loa='country_month', from_column='acled_sb_count')
            # .transform.ops.ln()
            .transform.missing.fill()
            .transform.missing.replace_na()
            )

            .with_column(Column('lr_acled_os', from_loa='country_month', from_column='acled_os_fat')
            # .transform.ops.ln()
            .transform.missing.fill()
            .transform.missing.replace_na()
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
        )

    def _add_wdi(queryset: Queryset) -> Queryset:
        """Economic indicators — pruned of redundant development proxies."""
        qs = (
            queryset
            # Migration & Refugees — strong conflict-relevant signal
            .with_column(
                Column("lr_wdi_sm_pop_netm", from_loa="country_year", from_column="wdi_sm_pop_netm")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_wdi_sm_pop_refg_or", from_loa="country_year", from_column="wdi_sm_pop_refg_or")
                .transform.missing.fill().transform.missing.replace_na()
            )
            # Aid & Military — institutional fragility signals
            .with_column(
                Column("lr_wdi_dt_oda_odat_pc_zs", from_loa="country_year", from_column="wdi_dt_oda_odat_pc_zs")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_wdi_ms_mil_xpnd_gd_zs", from_loa="country_year", from_column="wdi_ms_mil_xpnd_gd_zs")
                .transform.missing.fill().transform.missing.replace_na()
            )
            # Demographics — demographic pressure
            .with_column(
                Column("lr_wdi_sp_pop_grow", from_loa="country_year", from_column="wdi_sp_pop_grow")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_wdi_sp_urb_totl_in_zs", from_loa="country_year", from_column="wdi_sp_urb_totl_in_zs")
                .transform.missing.fill().transform.missing.replace_na()
            )
            # Health — state capacity / deprivation proxies
            .with_column(
                Column("lr_wdi_sp_dyn_imrt_fe_in", from_loa="country_year", from_column="wdi_sp_dyn_imrt_fe_in")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_wdi_sh_sta_maln_zs", from_loa="country_year", from_column="wdi_sh_sta_maln_zs")
                .transform.missing.fill().transform.missing.replace_na()
            )
        )

        # Conditionally include near-static economic structure features
        # These differentiate countries cross-sectionally but provide minimal
        # temporal signal. Keep without static covs (implicit country awareness),
        # remove with static covs (redundant).
        if not USE_STATIC_COVS:
            qs = (
                qs
                .with_column(
                    Column("lr_wdi_ny_gdp_mktp_kd", from_loa="country_year", from_column="wdi_ny_gdp_mktp_kd")
                    .transform.missing.fill().transform.missing.replace_na()
                )
                .with_column(
                    Column("lr_wdi_nv_agr_totl_kn", from_loa="country_year", from_column="wdi_nv_agr_totl_kn")
                    .transform.missing.fill().transform.missing.replace_na()
                )
            )

        return qs

    def _add_vdem(queryset: Queryset) -> Queryset:
        """
        Governance indicators — pruned of:
        - diagacc (redundant with horacc+veracc)
        - exlgender, exlpol (weakest exclusion dimensions, overlapping)
        - eqprotec (redundant with eqdr)
        - genpp (weakest conflict predictor)
        - divparctrl (sparse in authoritarian states)
        """
        return (
            queryset
            # Accountability — kept two most distinct dimensions
            .with_column(
                Column("lr_vdem_v2x_horacc", from_loa="country_year", from_column="vdem_v2x_horacc")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_vdem_v2x_veracc", from_loa="country_year", from_column="vdem_v2x_veracc")
                .transform.missing.fill().transform.missing.replace_na()
            )
            # Clientelism & Corruption — distinct governance failures
            .with_column(
                Column("lr_vdem_v2xnp_client", from_loa="country_year", from_column="vdem_v2xnp_client")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_vdem_v2xnp_regcorr", from_loa="country_year", from_column="vdem_v2xnp_regcorr")
                .transform.missing.fill().transform.missing.replace_na()
            )
            # Political Exclusion — kept two strongest conflict predictors
            .with_column(
                Column("lr_vdem_v2xpe_exlgeo", from_loa="country_year", from_column="vdem_v2xpe_exlgeo")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_vdem_v2xpe_exlsocgr", from_loa="country_year", from_column="vdem_v2xpe_exlsocgr")
                .transform.missing.fill().transform.missing.replace_na()
            )
            # Executive type — direct conflict-relevant institutional signal
            .with_column(
                Column("lr_vdem_v2x_ex_party", from_loa="country_year", from_column="vdem_v2x_ex_party")
                .transform.missing.fill().transform.missing.replace_na()
            )
            .with_column(
                Column("lr_vdem_v2x_ex_military", from_loa="country_year", from_column="vdem_v2x_ex_military")
                .transform.missing.fill().transform.missing.replace_na()
            )
            # Egalitarian governance — kept strongest grievance mechanism
            .with_column(
                Column("lr_vdem_v2xeg_eqdr", from_loa="country_year", from_column="vdem_v2xeg_eqdr")
                .transform.missing.fill().transform.missing.replace_na()
            )
            # Civil liberties — direct state violence/repression signals
            .with_column(
                Column("lr_vdem_v2xcl_prpty", from_loa="country_year", from_column="vdem_v2xcl_prpty")
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

    def _add_conflict_memory(queryset: Queryset) -> Queryset:
        """
        Exponential decay features: time_since(ged >= threshold) → decay(24).
        Values ∈ [0,1]. Update monthly. Encode conflict regime without level-leakage.
        Threshold tiers (5/100/500) provide separate onset/escalation/peak signals.
        """
        return (
            queryset
            # Own-country decay — state-based
            .with_column(Column('lr_decay_ged_sb_5', from_loa='country_month', from_column='ged_sb_best_sum_nokgi')
                .transform.missing.replace_na()
                .transform.bool.gte(5)
                .transform.temporal.time_since()
                .transform.temporal.decay(24)
                .transform.missing.replace_na()
            )
            .with_column(Column('lr_decay_ged_sb_100', from_loa='country_month', from_column='ged_sb_best_sum_nokgi')
                .transform.missing.replace_na()
                .transform.bool.gte(100)
                .transform.temporal.time_since()
                .transform.temporal.decay(24)
                .transform.missing.replace_na()
            )
            .with_column(Column('lr_decay_ged_sb_500', from_loa='country_month', from_column='ged_sb_best_sum_nokgi')
                .transform.missing.replace_na()
                .transform.bool.gte(500)
                .transform.temporal.time_since()
                .transform.temporal.decay(24)
                .transform.missing.replace_na()
            )
            .with_column(Column('lr_decay_ged_os_5', from_loa='country_month', from_column='ged_os_best_sum_nokgi')
                .transform.missing.replace_na()
                .transform.bool.gte(5)
                .transform.temporal.time_since()
                .transform.temporal.decay(24)
                .transform.missing.replace_na()
            )
            .with_column(Column('lr_decay_ged_os_100', from_loa='country_month', from_column='ged_os_best_sum_nokgi')
                .transform.missing.replace_na()
                .transform.bool.gte(100)
                .transform.temporal.time_since()
                .transform.temporal.decay(24)
                .transform.missing.replace_na()
            )
            .with_column(Column('lr_decay_ged_ns_5', from_loa='country_month', from_column='ged_ns_best_sum_nokgi')
                .transform.missing.replace_na()
                .transform.bool.gte(5)
                .transform.temporal.time_since()
                .transform.temporal.decay(24)
                .transform.missing.replace_na()
            )
            .with_column(Column('lr_decay_ged_ns_100', from_loa='country_month', from_column='ged_ns_best_sum_nokgi')
                .transform.missing.replace_na()
                .transform.bool.gte(100)
                .transform.temporal.time_since()
                .transform.temporal.decay(24)
                .transform.missing.replace_na()
            )
            .with_column(Column('lr_decay_acled_sb_5', from_loa='country_month', from_column='acled_sb_fat')
                .transform.missing.replace_na()
                .transform.bool.gte(5)
                .transform.temporal.time_since()
                .transform.temporal.decay(24)
                .transform.missing.replace_na()
            )
            .with_column(Column('lr_decay_acled_os_5', from_loa='country_month', from_column='acled_os_fat')
                .transform.missing.replace_na()
                .transform.bool.gte(5)
                .transform.temporal.time_since()
                .transform.temporal.decay(24)
                .transform.missing.replace_na()
            )
            .with_column(Column('lr_decay_acled_ns_5', from_loa='country_month', from_column='acled_ns_fat')
                .transform.missing.replace_na()
                .transform.bool.gte(5)
                .transform.temporal.time_since()
                .transform.temporal.decay(24)
                .transform.missing.replace_na()
            )
            # Spatial lag decay — cross-border conflict memory
            .with_column(Column('lr_splag_1_decay_ged_sb_5', from_loa='country_month', from_column='ged_sb_best_sum_nokgi')
                .transform.missing.replace_na()
                .transform.bool.gte(5)
                .transform.temporal.time_since()
                .transform.temporal.decay(24)
                .transform.spatial.countrylag(1, 1, 0, 0)
                .transform.missing.replace_na()
            )
            .with_column(Column('lr_splag_1_decay_ged_os_5', from_loa='country_month', from_column='ged_os_best_sum_nokgi')
                .transform.missing.replace_na()
                .transform.bool.gte(5)
                .transform.temporal.time_since()
                .transform.temporal.decay(24)
                .transform.spatial.countrylag(1, 1, 0, 0)
                .transform.missing.replace_na()
            )
            .with_column(Column('lr_splag_1_decay_ged_ns_5', from_loa='country_month', from_column='ged_ns_best_sum_nokgi')
                .transform.missing.replace_na()
                .transform.bool.gte(5)
                .transform.temporal.time_since()
                .transform.temporal.decay(24)
                .transform.spatial.countrylag(1, 1, 0, 0)
                .transform.missing.replace_na()
            )
        )

    def _add_temporal_lags(queryset: Queryset) -> Queryset:
        """
        Explicit ln_ged temporal lags for TiDE (no recurrence — must see recent
        conflict trajectory as explicit input features, not reconstructed from
        hidden state). ln() compresses Syria-scale outliers without level-leakage.
        """
        return (
            queryset
            .with_column(Column('ln_ged_sb_tlag_1', from_loa='country_month', from_column='ged_sb_best_sum_nokgi')
                .transform.ops.ln()
                .transform.missing.fill()
                .transform.temporal.tlag(1)
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(Column('ln_ged_sb_tlag_2', from_loa='country_month', from_column='ged_sb_best_sum_nokgi')
                .transform.ops.ln()
                .transform.missing.fill()
                .transform.temporal.tlag(2)
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(Column('ln_ged_sb_tlag_3', from_loa='country_month', from_column='ged_sb_best_sum_nokgi')
                .transform.ops.ln()
                .transform.missing.fill()
                .transform.temporal.tlag(3)
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(Column('ln_ged_sb_tlag_4', from_loa='country_month', from_column='ged_sb_best_sum_nokgi')
                .transform.ops.ln()
                .transform.missing.fill()
                .transform.temporal.tlag(4)
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(Column('ln_ged_sb_tlag_5', from_loa='country_month', from_column='ged_sb_best_sum_nokgi')
                .transform.ops.ln()
                .transform.missing.fill()
                .transform.temporal.tlag(5)
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(Column('ln_ged_sb_tlag_6', from_loa='country_month', from_column='ged_sb_best_sum_nokgi')
                .transform.ops.ln()
                .transform.missing.fill()
                .transform.temporal.tlag(6)
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(Column('ln_ged_os_tlag_1', from_loa='country_month', from_column='ged_os_best_sum_nokgi')
                .transform.ops.ln()
                .transform.missing.fill()
                .transform.temporal.tlag(1)
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
        )

    def _add_topic(queryset: Queryset) -> Queryset:
        """
        News topic model stocks — monthly leading indicators not in conflict counts.
        theta4 ~ political violence discourse, theta2/5 ~ instability discourse.
        t-13 lag: year-prior baseline for seasonal/structural comparison.
        """
        return (
            queryset
            .with_column(Column('lr_topic_tokens_t1', from_loa='country_month', from_column='topic_tokens')
                .transform.missing.fill()
                .transform.missing.replace_na()
                .transform.temporal.tlag(1)
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(Column('lr_topic_tokens_t2', from_loa='country_month', from_column='topic_tokens')
                .transform.missing.fill()
                .transform.missing.replace_na()
                .transform.temporal.tlag(2)
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(Column('lr_topic_ste_theta4_stock_t1', from_loa='country_month', from_column='topic_ste_theta4_stock')
                .transform.missing.fill()
                .transform.missing.replace_na()
                .transform.temporal.tlag(1)
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(Column('lr_topic_ste_theta4_stock_t2', from_loa='country_month', from_column='topic_ste_theta4_stock')
                .transform.missing.fill()
                .transform.missing.replace_na()
                .transform.temporal.tlag(2)
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(Column('lr_topic_ste_theta4_stock_t13', from_loa='country_month', from_column='topic_ste_theta4_stock')
                .transform.missing.fill()
                .transform.missing.replace_na()
                .transform.temporal.tlag(13)
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(Column('lr_topic_ste_theta2_stock_t1', from_loa='country_month', from_column='topic_ste_theta5_stock')
                .transform.missing.fill()
                .transform.missing.replace_na()
                .transform.temporal.tlag(1)
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(Column('lr_topic_ste_theta2_stock_t2', from_loa='country_month', from_column='topic_ste_theta5_stock')
                .transform.missing.fill()
                .transform.missing.replace_na()
                .transform.temporal.tlag(2)
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(Column('lr_topic_ste_theta2_stock_t13', from_loa='country_month', from_column='topic_ste_theta5_stock')
                .transform.missing.fill()
                .transform.missing.replace_na()
                .transform.temporal.tlag(13)
                .transform.missing.fill()
                .transform.missing.replace_na()
            )
            .with_column(Column('lr_topic_ste_theta4_stock_t1_splag', from_loa='country_month', from_column='topic_ste_theta4_stock')
                .transform.missing.fill()
                .transform.missing.replace_na()
                .transform.temporal.tlag(13)
                .transform.missing.fill()
                .transform.spatial.countrylag(1, 1, 0, 0)
                .transform.missing.replace_na()
            )
            .with_column(Column('lr_topic_ste_theta2_stock_t1_splag', from_loa='country_month', from_column='topic_ste_theta5_stock')
                .transform.missing.fill()
                .transform.missing.replace_na()
                .transform.temporal.tlag(13)
                .transform.missing.fill()
                .transform.spatial.countrylag(1, 1, 0, 0)
                .transform.missing.replace_na()
            )
        )

    queryset = Queryset(f"{model_name}", "country_month")
    return _add_topic(_add_temporal_lags(_add_conflict_memory(_add_vdem(_add_wdi(_add_conflict_history(queryset))))))