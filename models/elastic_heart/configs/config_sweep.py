def get_sweep_config():
    """
    meow
    """
    sweep_config = {
        "method": "bayes",
        "name": "elastic_heart_tsmixer_spotlight_v20_msle",
        "early_terminate": {"type": "hyperband", "min_iter": 30, "eta": 2},
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        # 36 = 1× output_chunk_length (minimum sensible context).
        # 36 = minimal (1× output). 48 = 4yr context for 3yr forecast.
        # 72 = production default (2× output — sees a full conflict cycle before window).
        "input_chunk_length": {"values": [36, 48, 72]},
        "output_chunk_length": {"values": [36]},
        "output_chunk_shift": {"values": [0]},
        "random_state": {"values": [67]},
        "mc_dropout": {"values": [False]},
        "optimizer_cls": {"values": ["AdamW", "RAdam"]},
        "num_samples": {"values": [1]},
        "n_jobs": {"values": [-1]},
        # ==============================================================================
        # TRAINING
        # ==============================================================================
        "batch_size": {"values": [64]},
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [50]},
        "early_stopping_min_delta": {"values": [0.0001]},
        "force_reset": {"values": [True]},
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        # TSMixer is a pure-MLP architecture — simpler gradient landscape
        # than TiDE/Transformers, so slightly wider LR range is safe.
        # [5e-5, 1e-3]: anchor 3e-4 sits at ~66th percentile on log scale — symmetric enough
        # for Bayes to explore both sides. Previous [3e-5, 5e-4] put anchor at 85th
        # percentile: Bayes converged high immediately, never tried smaller LRs.
        "lr": {
            "distribution": "log_uniform_values",
            "min": 5e-5,
            "max": 1e-3,
        },
        "weight_decay": {"values": [0, 1e-5, 1e-4]},
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["CosineAnnealingWarmRestarts"]},
        "lr_scheduler_T_0": {"values": [30]},
        "lr_scheduler_T_mult": {"values": [2]},
        "lr_scheduler_eta_min": {"values": [1e-6]},
        # Max per-cell pointwise gradient = w(y)×tanh ≤ 4.3 (alpha=0.35).
        # clip=3.0 trims only the most extreme event cells. clip=5.0 dropped —
        # effectively disables clipping and v19 showed clip=5 + alpha>0.22 → 2-8× overprediction.
        "gradient_clip_val": {"values": [2.0, 3.0]},
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
                        "lr_ged_ns", 
                        "lr_ged_os",
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
        # TSMIXER ARCHITECTURE
        # ==============================================================================
        # num_blocks: Number of mixer blocks (time-mix + feature-mix pairs).
        # TSMixer paper uses 2 for small datasets. 3 gives more capacity
        # but risks overfitting on ~200 series.
        "num_blocks": {"values": [2, 3]},
        # hidden_size: Bottleneck dim in feature mixing. 64 is too tight
        # for 59 input channels — destroys cross-feature information.
        "hidden_size": {"values": [128]},
        # ff_size: Expansion dim. Must be 2-4x hidden_size for meaningful
        # cross-feature interaction. v1's 128/128 = 1:1 ratio was effectively
        # a residual identity (no expansion). 256=2x, 512=4x.
        "ff_size": {"values": [256, 512]},
        # normalize_before: Pre-norm (True) is more training-stable,
        # post-norm (False) can give slightly better final performance.
        "normalize_before": {"values": [True, False]},
        # GELU only — smoother gradients than ReLU, no dead neurons.
        # v1 best run picked GELU.
        "activation": {"values": ["GELU"]},
        # norm_type: LayerNorm is standard for MLP-mixers.
        # TimeBatchNorm2d is the TSMixer-specific variant.
        "norm_type": {"values": ["LayerNorm"]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        # Dropout: TSMixer has 3-5 MonteCarloDropout layers per block.
        # v1 best landed at 0.223 (86% of range). Raise floor since
        # Bayes clearly wants higher dropout for ~200 series.
        "dropout": {"values": [0.15, 0.25, 0.35]},
        "use_static_covariates": {"values": [True]},
        "use_reversible_instance_norm": {"values": [True, False]},
        # ==============================================================================
        # LOSS FUNCTION: SpotlightLoss
        # v20: truth-only 1+log_cosh(alpha*|y|) weight + balanced mean + multi-res spectral
        # TV removed — spectral is a strict superset (oscillation + drift + seasonality + phase)
        # ==============================================================================
        "loss_function": {"values": ["SpotlightLoss"]},
        # ── alpha (truth-only spotlight scale) ───────────────────────────────────────
        # 1+log_cosh(alpha*|y|) — truncated-inverse-density weight (Liu & Lin 2022;
        # Yang et al. 2021 LDS). No pred-side weight — gradient bounded by w(y)×tanh.
        # Weight at max UCDP (asinh≈11.5):
        #   alpha=0.15 → ≈2.1×   alpha=0.25 → ≈3.2×   alpha=0.35 → ≈4.3×
        # GRADIENT BUDGET: alpha scales pointwise gradient magnitude. Capped at 0.35
        # (4.3× max weight) so the pointwise-to-spectral gradient ratio stays in
        # [2:1, 6:1] across the full delta range. alpha=0.5 was 6.1× — starved
        # spectral of gradient budget at low delta, causing it to be ignored.
        # Test run anchor: alpha=0.2, delta=0.15 → balanced.
        "alpha": {
            "distribution": "uniform",
            "min": 0.15,
            "max": 0.35,
        },
        "non_zero_threshold": {"values": [0.88]},  # asinh(1) ≈ 0.88, i.e. ≥1 battle-related death
        # ── delta (multi-resolution spectral weight) ─────────────────────────────────
        # Spectral L1-magnitude matching (n_fft=6,12,24). Phase-insensitive by
        # the Fourier shift theorem: onset 1-mo early → ~zero spectral penalty.
        # n_fft=12 bin 1 = 12-month annual cycle — directly penalises missing seasonality.
        # n_fft=24 catches slow monotonic drift (smooth hockey sticks TV couldn't detect).
        # GRADIENT BUDGET: STFT accumulates ~48 gradient paths per time step across
        # 3 resolutions (8+14+26 bins×frames) vs 1 for pointwise. After .mean()
        # normalisation, spectral gradient norm is ~5-10× pointwise before delta.
        #   delta=0.08 → spectral ≈10-15% of total gradient (light regularisation)
        #   delta=0.15 → spectral ≈20-30% of total gradient (test run anchor)
        #   delta=0.25 → spectral ≈35-45% of total gradient (heavy temporal shaping)
        # Floor at 0.08 so spectral is never noise. Cap at 0.25 so pointwise
        # accuracy isn't starved — the model still needs to get cell values right.
        "delta": {
            "distribution": "uniform",
            "min": 0.08,
            "max": 0.25,
        },
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config