def get_sweep_config():
    """
    meow
    """
    sweep_config = {
        "method": "bayes",
        "name": "elastic_heart_tsmixer_spotlight_v12_msle",
        "early_terminate": {"type": "hyperband", "min_iter": 30, "eta": 2},
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        "input_chunk_length": {"values": [36]},
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
        "early_stopping_patience": {"values": [40]},
        "early_stopping_min_delta": {"values": [0.0001]},
        "force_reset": {"values": [True]},
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        # TSMixer is a pure-MLP architecture — simpler gradient landscape
        # than TiDE/Transformers, so slightly wider LR range is safe.
        "lr": {
            "distribution": "log_uniform_values",
            "min": 3e-5,
            "max": 5e-4,
        },
        "weight_decay": {"values": [0, 1e-5, 5e-5]},
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["CosineAnnealingWarmRestarts"]},
        "lr_scheduler_T_0": {"values": [30]},
        "lr_scheduler_T_mult": {"values": [2]},
        "lr_scheduler_eta_min": {"values": [1e-6]},
        # Batch-normalised magnitude weights keep gradient scale bounded,
        # so we can afford a wider clip range.
        "gradient_clip_val": {"values": [2.0, 3.0, 5.0]},
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
        # v11: log-cosh magnitude weights + tanh² bounded error amplification
        # ==============================================================================
        "loss_function": {"values": ["SpotlightLoss"]},
        # ── alpha (log-cosh magnitude rate) ──────────
        # 1 + log(cosh(alpha * |y|)) grows linearly for large |y|, so
        # higher alpha is safe — no exponential blow-up.  At 1.0 the
        # 50k-fatality cell gets ~12× weight vs peace after batch norm.
        "alpha": {
            "distribution": "uniform",
            "min": 0.2,
            "max": 0.5,
        },
        "non_zero_threshold": {"values": [0.88]},  # asinh(1) ≈ 0.88, i.e. ≥1 battle-related death
        # ── gamma (temporal gradient weight) ──────────
        "gamma": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 1.5,
        },
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config