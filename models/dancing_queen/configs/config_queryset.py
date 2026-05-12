from viewser import Queryset, Column
from views_pipeline_core.managers.model import ModelPathManager

model_name = ModelPathManager.get_model_name_from_path(__file__)


def generate():
    """
    Pruned queryset for conflict forecasting with TiDE + MagnitudeAwareHuberLoss.
    
    Feature Strategy (lean encoder, expressive decoder):
    ----------------------------------------------------
    1. Conflict History: GED counts + momentum (deltas) + ACLED event count
    2. Spatial Spillover: Neighboring country conflict (direct GED spatial lags)
    3. Governance (V-Dem): Pruned set — removed redundant accountability,
       exclusion, and equality indices (see removal log below)
    4. Economic Stress (WDI): Pruned set — removed redundant development
       proxies and near-static structural indicators
    
    Removed (11 features, always):
    - lr_acled_sb: Redundant with lr_ged_sb (same phenomenon, different source)
    - lr_acled_os: Redundant with lr_ged_os (same phenomenon, different source)
    - lr_wdi_sh_sta_stnt_zs: Redundant with malnutrition (r>0.85), chronic not acute
    - lr_vdem_v2x_diagacc: Redundant with horacc+veracc, weakest independent signal
    - lr_vdem_v2xpe_exlgender: Weakest exclusion dimension for conflict, high r with exlsocgr
    - lr_vdem_v2xpe_exlpol: Overlaps with vertical accountability conceptually
    - lr_vdem_v2xeg_eqprotec: Redundant with eqdr (r>0.8), weaker conflict mechanism
    - lr_vdem_v2x_genpp: Weakest V-Dem conflict predictor, collinear with democracy cluster
    - lr_wdi_se_enr_prim_fm_zs: Sparse in conflict states, redundant development proxy
    - lr_wdi_sl_tlf_totl_fe_zs: No independent conflict mechanism, redundant with GDP+urban
    - lr_vdem_v2x_divparctrl: Skewed/missing in authoritarian states, weak conflict link
    
    Removed (2 features, conditional on static covariates):
    - lr_wdi_nv_agr_totl_kn: Near-static; cross-sectional signal redundant with static covs
    - lr_wdi_ny_gdp_mktp_kd: Near-static; better as static covariate than time-varying
    
    Feature Counts:
    - Conflict: 10 (4 GED/ACLED + 3 delta + 3 splag)
    - WDI: 10 (no static covs) / 8 (with static covs)
    - V-Dem: 12
    - Total: 32 features (no static covs) / 30 (with static covs) + 3 targets
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
            .with_column(
                Column('lr_acled_sb_count', from_loa='country_month', from_column='acled_sb_count')
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

    queryset = Queryset(f"{model_name}", "country_month")
    return _add_vdem(_add_wdi(_add_conflict_history(queryset)))