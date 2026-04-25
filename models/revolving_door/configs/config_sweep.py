
def get_sweep_config():
    """N-HiTS + PrismLoss v33 sweep configuration."""

    sweep_config = {
        "method": "bayes",
        "name": "revolving_door_nhits_prism_v11_msle",
        "early_terminate": {"type": "hyperband", "min_iter": 50, "eta": 2},  # 50 > CAWR T_0=30 — avoids terminating runs at the LR spike before they recover
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        "input_chunk_length": {"values": [48]},
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
        # N-HiTS paper uses LR ~1e-3 but SpotlightLoss cosh amplification
        # injects higher gradient variance — keep ceiling conservative.
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
        # T_mult=2: progressive lengthening (30→60→120) lets the model
        # settle into finer optima in later cycles.
        "lr_scheduler_T_mult": {"values": [2]},
        "lr_scheduler_eta_min": {"values": [1e-6]},
        # With alpha <= 0.20, max cosh contribution at asinh(10000)=9.9 is
        # cosh(0.20*9.9)≈3.8. Tanh gradient is bounded at ±1. Effective max
        # gradient ≈ 3.8 — clip of 2.0 is sufficient. 5.0 provides no protection.
        "gradient_clip_val": {"values": [10.0]},  # MSE in log1p: max gradient ≈ 11 (log1p(50000)). clip=10 trims only extreme outliers.
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
        # N-HiTS ARCHITECTURE
        # ==============================================================================
        # 3 stacks: coarse (annual) + intermediate (quarterly) + fine (monthly).
        # Pooling kernel rationale (icl=48):
        #   kernel=12 → 48/12 = 4 annual groups  (coarse multi-year trends)
        #   kernel=4  → 48/4  = 12 quarterly groups (conflict-cycle patterns)
        #   kernel=1  → 48 raw monthly steps       (fine stack always unpooled)
        # n_freq_downsample rationale (ocl=36, independent of icl):
        #   fd=6 → 36/6 = 6 basis functions  (slow structural trends, semi-annual)
        #   fd=3 → 36/3 = 12 basis functions (quarterly detail)
        #   fd=1 → 36/1 = 36 basis functions (full monthly)
        "num_stacks": {"values": [3]},
        "pooling_kernel_sizes": {"values": [[[12], [4], [1]]]},
        "n_freq_downsample": {"values": [[[12], [3], [1]]]},
        "max_pool_1d": {"values": [False]},
        "activation": {"values": ["GELU"]},
        # num_blocks/num_layers/layer_widths: Darts defaults are 1 block, 2 layers,
        # 512 width per stack. Capacity sweep: 2 blocks gives each stack more
        # representational power. layer_widths=256 confirmed too small; 512/1024 sweep.
        "num_blocks": {"values": [1, 2]},
        "num_layers": {"values": [3]},
        "layer_widths": {"values": [512, 1024]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        "dropout": {"values": [0.15, 0.25]},
        "use_static_covariates": {"values": [True]},
        # RevIN off: log1p space, peace series have σ≈0 → RevIN divides by σ → NaN.
        # OOD instability was from sinh inversion (asinh scaler), not from RevIN absence.
        # With LogTransform + expm1 inversion, outputs are bounded without RevIN.
        "use_reversible_instance_norm": {"values": [False]},
        # ==============================================================================
        # LOSS FUNCTION: PrismLoss
        # ==============================================================================
        # N-HiTS is feedforward — no BPTT attenuation. Same SpotlightLoss
        # parameter philosophy as smol_cat (TiDE): alpha can be used freely
        # for cosh magnitude weighting.
        "loss_function": {"values": ["PrismLoss"]},
        # alpha removed in PrismLoss v33. MSE in log1p = MSLE exactly.
        # log_cosh+alpha was causing 10-16× gradient deficit via tanh saturation.
        "non_zero_threshold": {"values": [0.693]},  # log1p(1) ≈ 0.693, i.e. ≥1 battle-related death
        # ── delta (multi-resolution spectral weight) ─────────────────────────────────
        # Spectral log_cosh(|S_pred| - |S_true|) at n_fft=6,12,24. DC bin masked.
        # MSE pointwise gradient scales as e (up to ~11). Spectral bounded at tanh ≤ 1.
        # Ratio ~5-10:1 pointwise/spectral before delta. Floor 0.05 = spectral not noise.
        "delta": {
            "distribution": "uniform",
            "min": 0.05,
            "max": 0.20,
        },
        # dual_mean=False: training loss = MSLE exactly. True caused false minimum
        # (train_loss=0.28 vs MSLE_val=0.80 with different fixed points).
        "dual_mean": {"values": [False]},
        "event_weight": {
            "values": [0.50], # not used
        },
        # ModelCatalog builds the encoder dict from this flag at model-build
        # time, selecting functions based on config["level"] — JSON-safe.
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config