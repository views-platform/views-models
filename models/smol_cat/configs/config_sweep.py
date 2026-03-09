def get_sweep_config():
    """
    TiDE Hyperparameter Sweep Configuration - MAATLoss
    ===================================================

    Strategy: Asinh-space regression with magnitude-aligned asymmetric temporal loss
    ---------------------------------------------------------------------------------

    v22: MAAT-Loss sweep — Country-Month (CM) level forecasting
    -------------------------------------------------------------
    Upgrades from JATLoss (v21) to MAATLoss — a 4-component composite loss
    addressing all three failure modes (Drama Queen, Coward, Bad Timer):

      1. Magnitude-recovering weighted Huber: cosh(α·max(|ŷ|,|y|)) Jacobian
         weight with Huber base — restores raw-scale gradient sensitivity.
      2. CDF temporal alignment (Cramér distance): penalizes cumulative
         temporal offsets, not just pointwise errors.
      3. Temporal derivative penalty: cosine-similarity on first-differences
         weighted by true change magnitude — enforces onset/offset direction.
      4. Asymmetric soft-focal hurdle: sigmoid-relaxed peace/conflict boundary
         with focal cross-entropy, β_FN > β_FP for asymmetric FN/FP tradeoff.

    Advantages over JATLoss:
    - CDF alignment directly penalizes "right magnitude, wrong timing"
    - Soft-focal hurdle provides explicit classification signal at the
      peace/conflict boundary (vs. JATLoss implicit via asymmetric MSE)
    - Derivative penalty catches onset/offset direction errors
    - Magnitude weight cap (w_max) prevents gradient explosion without
      relying solely on percentile clamping

    Country-Month (CM) level notes:
    - ~200 countries × 36-month horizon → ~7,200 cells per output window
    - Denser conflict signal per series than PGM (~60% have ≥1 conflict month)
    - Larger per-series magnitudes (country aggregates) → higher asinh values
    - Lower α (0.3–0.6) sufficient since country-level asinh values are larger
    - Stronger CDF alignment (λ_cdf up to 0.3) since country time series
      are smoother and temporal structure is more coherent
    - Moderate hurdle weight — country series have clearer peace/conflict split

    Features: 37 (6 conflict + 13 WDI + 18 V-Dem, no topics).
    Input tensor per window: 37 × 36 = 1,332 values.
    Training windows: ~16.5K.

    Architecture notes:
    - RevIN disabled: Jacobian weighting handles magnitude natively
    - mc_dropout=False: Deterministic inference
    - Position encoder: relative only (integer month_id indexing)
    """
    sweep_config = {
        "method": "bayes",
        "name": "smol_cat_tide_maat_v22_cm_msle",
        "early_terminate": {"type": "hyperband", "min_iter": 30, "eta": 2},
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        "input_chunk_length": {"values": [36]},
        "output_chunk_shift": {"values": [0]},
        "random_state": {"values": [67]},
        "output_chunk_length": {"values": [36]},
        "optimizer_cls": {"values": ["Adam"]},
        "mc_dropout": {"values": [False]},
        "num_samples": {"values": [1]},
        "n_jobs": {"values": [-1]},
        # ==============================================================================
        # TRAINING
        # ==============================================================================
        # Batch size: Fixed at 64. MAAT magnitude weighting + hurdle produces
        # heterogeneous per-sample gradients — need enough samples for stable
        # batch gradient estimates. Country-level has fewer series than PGM,
        # so 64 is appropriate.
        "batch_size": {"values": [64]},
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [40]},
        "early_stopping_min_delta": {"values": [0.0001]},
        "force_reset": {"values": [True]},
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        # LR: MAAT has more gradient sources (4 components) — slightly wider
        # range to accommodate different component balance regimes.
        "lr": {
            "distribution": "log_uniform_values",
            "min": 3e-5,
            "max": 3e-4,
        },
        "weight_decay": {"values": [5e-6]},
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["CosineAnnealingWarmRestarts"]},
        "lr_scheduler_T_0": {"values": [30]},
        "lr_scheduler_T_mult": {"values": [2]},
        "lr_scheduler_eta_min": {"values": [1e-6]},
        # MAAT: cosh weight is capped at w_max (default 100), and Huber base
        # limits gradient growth. Lower clip than raw JATLoss needed.
        "gradient_clip_val": {"values": [1.0, 2.0, 3.0]},
        # ==============================================================================
        # SCALING
        # ==============================================================================
        "feature_scaler": {"values": [None]},
        "target_scaler": {"values": ["AsinhTransform"]},
        "feature_scaler_map": {
            "values": [
                {
                    "AsinhTransform": [
                        "lr_wdi_sm_pop_refg_or",
                        "lr_wdi_ny_gdp_mktp_kd",
                        "lr_wdi_nv_agr_totl_kn",
                        "lr_splag_1_ged_sb",
                        "lr_splag_1_ged_ns",
                        "lr_splag_1_ged_os",
                    ],
                    "StandardScaler": [
                        "lr_ged_sb_delta",
                        "lr_ged_ns_delta",
                        "lr_ged_os_delta",
                        "lr_wdi_sm_pop_netm",
                        "lr_wdi_dt_oda_odat_pc_zs",
                        "lr_wdi_sp_pop_grow",
                        "lr_wdi_ms_mil_xpnd_gd_zs",
                        "lr_wdi_sp_dyn_imrt_fe_in",
                        "lr_wdi_sh_sta_stnt_zs",
                        "lr_wdi_sh_sta_maln_zs",
                    ],
                    "MinMaxScaler": [
                        "lr_wdi_sl_tlf_totl_fe_zs",
                        "lr_wdi_se_enr_prim_fm_zs",
                        "lr_wdi_sp_urb_totl_in_zs",
                        "lr_vdem_v2x_horacc",
                        "lr_vdem_v2x_veracc",
                        "lr_vdem_v2x_diagacc",
                        "lr_vdem_v2xnp_client",
                        "lr_vdem_v2xnp_regcorr",
                        "lr_vdem_v2xpe_exlpol",
                        "lr_vdem_v2xpe_exlgeo",
                        "lr_vdem_v2xpe_exlgender",
                        "lr_vdem_v2xpe_exlsocgr",
                        "lr_vdem_v2x_divparctrl",
                        "lr_vdem_v2x_ex_party",
                        "lr_vdem_v2x_ex_military",
                        "lr_vdem_v2x_genpp",
                        "lr_vdem_v2xeg_eqdr",
                        "lr_vdem_v2xcl_prpty",
                        "lr_vdem_v2xeg_eqprotec",
                        "lr_vdem_v2xcl_dmove",
                        "lr_vdem_v2x_clphy",
                    ],
                }
            ]
        },
        # ==============================================================================
        # TiDE ARCHITECTURE
        # ==============================================================================
        # Country-month: fewer series (~200) but richer temporal structure.
        # Need sufficient capacity to model diverse country trajectories.
        "num_encoder_layers": {"values": [1]},
        # Decoder layers: 2 or 3. Country series are smoother than PGM —
        # 2 layers may suffice, but 3 gives more capacity for diverse patterns.
        "num_decoder_layers": {"values": [2, 3]},
        "decoder_output_dim": {"values": [64]},
        # hidden_size: SWEPT. Country-level needs capacity for ~200 diverse
        # trajectories. 256 is minimum viable, 512 gives headroom.
        "hidden_size": {"values": [256, 512]},
        # temporal_width: Country series have stronger annual cycles.
        # 4 (paper default) vs 12 (annual cycle match).
        "temporal_width_past": {"values": [4, 12]},
        "temporal_width_future": {"values": [36, 48]},
        "temporal_decoder_hidden": {"values": [256]},
        "temporal_hidden_size_past": {"values": [128]},
        "temporal_hidden_size_future": {"values": [128]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        "use_layer_norm": {"values": [True]},
        # Dropout: Country-level has fewer training windows per series.
        # Slightly higher dropout ceiling to prevent overfitting on ~200 series.
        "dropout": {
            "distribution": "uniform",
            "min": 0.05,
            "max": 0.20,
        },
        "use_static_covariates": {"values": [True]},
        "use_reversible_instance_norm": {"values": [False]},
        # ==============================================================================
        # LOSS FUNCTION: MAATLoss (Magnitude-Aligned Asymmetric Temporal Loss)
        # ==============================================================================
        # Four-component loss for asinh-transformed zero-inflated data:
        #   A. Magnitude-recovering weighted Huber (cosh Jacobian weight)
        #   B1. CDF temporal alignment (Cramér distance on cumulative sums)
        #   B2. Temporal derivative penalty (cosine-sim on first-differences)
        #   C. Asymmetric soft-focal classification (hurdle)
        #
        # Stability: w_max caps Jacobian weight. 99.9th-percentile per-sample
        # clamp inside loss. gradient_clip_val ≥ 1.0 externally.
        "loss_function": {"values": ["MAATLoss"]},
        #
        # ── alpha (magnitude expansion rate) ──────────
        # Controls how aggressively cosh undoes asinh compression.
        # Country-level has higher asinh values (aggregated fatalities) →
        # lower α avoids excessive weighting at the country-aggregate tail.
        #   0.3 = conservative (cosh(0.3×9)=3.5× at asinh=9)
        #   0.5 = moderate (cosh(0.5×9)=45× at asinh=9)
        #   0.6 = aggressive (cosh(0.6×9)=132× at asinh=9)
        "alpha": {
            "distribution": "uniform",
            "min": 0.3,
            "max": 0.6,
        },
        # ── w_max (magnitude weight cap) ──────────────
        # Prevents gradient explosion at extreme tail values.
        # At α=0.5, asinh(4000)=9: cosh(4.5)≈45. Cap at 100 gives headroom.
        # Country aggregates can reach asinh(50000)≈11.5: cosh(5.75)≈157,
        # capped at 100. Fixed — not worth sweeping.
        "w_max": {"values": [100.0]},
        # ── huber_delta ───────────────────────────────
        # Huber transition point. Controls sensitivity to large residuals.
        # At δ=1: quadratic for |r|<1, linear for |r|>1.
        # Country-level residuals tend to be larger → slightly higher δ OK.
        "huber_delta": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1.5,
        },
        # ── tau (peace/conflict threshold) ────────────
        # Threshold in asinh space for the soft hurdle classifier.
        # asinh(1)≈0.88 (at least 1 fatality) is the natural boundary.
        # Country-level could use slightly higher (asinh(5)≈2.3) since
        # country aggregates rarely have exactly 1 fatality.
        "tau": {"values": [0.88]},
        # ── kappa (sigmoid sharpness) ─────────────────
        # Controls how hard the soft peace/conflict boundary is.
        # Higher kappa → sharper sigmoid → more binary classification signal.
        # Fixed at 5.0: at τ=0.88, σ(5·(2.3-0.88))≈1.0 (clear conflict).
        "kappa": {"values": [5.0]},
        # ── beta_fn / beta_fp (FN/FP asymmetry) ──────
        # Controls the false negative vs false positive tradeoff in the
        # hurdle component. ρ = beta_fn / beta_fp is the FN/FP ratio.
        # Country-level: missing a country-level conflict onset is critical
        # for early warning → moderate-to-high ρ.
        #   ρ=2: FN penalty 2× FP
        #   ρ=3: FN penalty 3× FP (recommended)
        #   ρ=5: FN penalty 5× FP (aggressive, risk Drama Queen)
        "beta_fn": {
            "distribution": "uniform",
            "min": 2.0,
            "max": 5.0,
        },
        "beta_fp": {"values": [1.0]},
        # ── focal_gamma ───────────────────────────────
        # Focal exponent for easy-example down-weighting in the hurdle.
        # γ=2.0 is the standard Lin et al. (2017) recommendation.
        # Country-level has better class separation → γ=2 is appropriate.
        "focal_gamma": {"values": [2.0]},
        # ── lambda_reg ────────────────────────────────
        # Weight for the magnitude-recovering regression component.
        # Fixed at 1.0 as the reference scale.
        "lambda_reg": {"values": [1.0]},
        # ── lambda_cdf ────────────────────────────────
        # Weight for CDF temporal alignment.
        # Country time series are smoother with more coherent temporal
        # structure → CDF alignment is more useful at CM than PGM.
        #   0.05 = light temporal guidance
        #   0.1  = moderate (good starting point)
        #   0.3  = strong temporal constraint
        "lambda_cdf": {
            "distribution": "log_uniform_values",
            "min": 0.05,
            "max": 0.3,
        },
        # ── lambda_deriv ──────────────────────────────
        # Weight for temporal derivative penalty.
        # Lighter touch than CDF — catches onset/offset direction errors.
        "lambda_deriv": {
            "distribution": "log_uniform_values",
            "min": 0.01,
            "max": 0.1,
        },
        # ── lambda_hurdle ─────────────────────────────
        # Weight for the asymmetric soft-focal classification component.
        # Primary anti-Coward mechanism. Too high risks Drama Queen if
        # beta_fn is also high.
        #   0.3 = moderate classification pressure
        #   0.5 = balanced (recommended)
        #   1.0 = strong classification pressure
        "lambda_hurdle": {
            "distribution": "uniform",
            "min": 0.3,
            "max": 1.0,
        },
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        "add_encoders": {
            "values": [
                {
                    "position": {"past": ["relative"], "future": ["relative"]},
                }
            ]
        },
    }

    sweep_config["parameters"] = parameters
    return sweep_config