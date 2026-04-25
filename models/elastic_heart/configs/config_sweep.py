def get_sweep_config():
    """
    meow
    """
    sweep_config = {
        "method": "bayes",
        "name": "elastic_heart_tsmixer_prism_v21_msle",
        "early_terminate": {"type": "hyperband", "min_iter": 30, "eta": 2},  # Rungs at 30,60,120. With patience=50, most runs die ~epoch 80 (before restart at 90). Hyperband fires twice before early stopping. eta=2: 50% killed per rung.
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
        "input_chunk_length": {"values": [36, 48]},
        "output_chunk_length": {"values": [36]},
        "output_chunk_shift": {"values": [0]},
        "random_state": {"values": [67]},
        "mc_dropout": {"values": [False]},
        "optimizer_cls": {"values": ["AdamW"]},
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
        "weight_decay": {"values": [0, 1e-4]},
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
        "gradient_clip_val": {"values": [10]},
        # ==============================================================================
        # SCALING
        # ==============================================================================
        "feature_scaler": {"values": [None]},
        "target_scaler": {"values": ["LogTransform"]},  # log1p(x): model operates in MSLE space. MSE in log1p = MSLE exactly.
        "feature_scaler_map": {
            "values": [
                {
                    "LogTransform": [
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
        # hidden_size: 128 = marginal for 43 features + conflict dynamics.
        # 256 doubles capacity. Both valid — Bayes picks the winner.
        "hidden_size": {"values": [128, 256]},
        # ff_size: 4× hidden with GELU (no gating, full expansion).
        # 512=4× hidden=128, 1024=4× hidden=256. Degenerate pairings avoided:
        # ff=256 bottlenecks hidden=256; ff=1024 is wasteful for hidden=128
        # but Bayes will learn to avoid it.
        "ff_size": {"values": [512, 1024]},
        # normalize_before: Fixed True. Pre-norm is more training-stable.
        # Post-norm marginal gain doesn't justify doubling architecture combos.
        "normalize_before": {"values": [True]},
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
        # Dropout: 0.35 rarely wins for conflict data. Two values cover the range.
        "dropout": {"values": [0.15, 0.25]},
        "use_static_covariates": {"values": [True]},
        # RevIN off permanently. In log1p space, peace series have σ≈0 —
        # RevIN divides by σ → NaN/inf. Also destroys the balanced-budget
        # design of PrismLoss.
        "use_reversible_instance_norm": {"values": [False]},
        # ==============================================================================
        # LOSS FUNCTION: PrismLoss
        # v20: truth-only 1+log_cosh(alpha*|y|) weight + balanced mean + multi-res spectral
        # TV removed — spectral is a strict superset (oscillation + drift + seasonality + phase)
        # ==============================================================================
        "loss_function": {"values": ["PrismLoss"]},
        # alpha removed in PrismLoss v33. Base loss is plain MSE in log1p space
        # (= MSLE directly). No per-cell weighting — MSE's quadratic scaling
        # naturally upweights large errors (events). alpha was causing
        # 10-16× gradient deficit via log_cosh saturation.
        "non_zero_threshold": {"values": [0.693]},  # log1p(1) ≈ 0.693, i.e. ≥1 battle-related death
        # ── delta (multi-resolution spectral weight) ─────────────────────────────────
        # Spectral log_cosh(|S_pred| - |S_true|) at n_fft=6,12,24. Phase-insensitive.
        # n_fft=12 bin 1 = 12-month annual cycle. DC bin masked.
        # MSE pointwise gradient scales as e (up to ~11 for log1p(50000)).
        # Spectral log_cosh gradient bounded at tanh ≤ 1. Ratio ~5-10:1 before delta.
        #   delta=0.05 → spectral ~2-4% of gradient (very light)
        #   delta=0.10 → light (~4-8%)
        #   delta=0.20 → moderate (~8-15%)
        "delta": {
            "distribution": "uniform",
            "min": 0.05,
            "max": 0.20,
        },
        # dual_mean: False = plain per-cell mean = training loss is MSLE exactly.
        # True was causing a false minimum (train_loss=0.28 vs MSLE_val=0.80
        # due to balanced-mean and MSLE having different fixed points).
        "event_weight": {
            "values": [0.50], # not used
        },
        "dual_mean": {"values": [False]},
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config