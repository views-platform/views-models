def get_sweep_config():
    """BlockRNN (GRU) + PrismLoss v33 sweep configuration."""
    sweep_config = {
        "method": "bayes",
        "name": "dancing_queen_blockrnn_prism_v7_msle",
        "early_terminate": {"type": "hyperband", "min_iter": 50, "eta": 2},
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        # 36 for BPTT safety, 48 for full context (Bayes decides).
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
        # BLOCKRNN ARCHITECTURE
        # ==============================================================================
        # GRU locked: update gate handles zero-heavy sequences better than
        # LSTM's forget gate (which aggressively zeros cell state).
        "rnn_type": {"values": ["GRU"]},
        # hidden_dim: 64 confirmed capacity-limited. 128 minimum viable, 256 adds headroom.
        "hidden_dim": {"values": [128, 256]},
        "n_rnn_layers": {"values": [1]},
        # hidden_fc_sizes: match hidden_dim. 128 is undersized as decoder for hidden=256.
        "hidden_fc_sizes": {"values": [[128], [256]]},
        # GELU: smoother gradients through the FC decoder.
        "activation": {"values": ["GELU"]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        "dropout": {"values": [0.15, 0.25]},
        "use_static_covariates": {"values": [True]},
        # RevIN off: log1p space, peace series have σ≈0 → RevIN divides by σ → NaN.
        "use_reversible_instance_norm": {"values": [False]},
        # ==============================================================================
        # LOSS FUNCTION: PrismLoss
        # ==============================================================================
        "loss_function": {"values": ["PrismLoss"]},
        # alpha removed in PrismLoss v33. MSE in log1p = MSLE exactly.
        # log_cosh+alpha was causing 10-16× gradient deficit via tanh saturation.
        # BPTT compounds gradients across steps but MSE gradient (e) is linear
        # and bounded — no special BPTT safety constraint needed.
        "non_zero_threshold": {"values": [0.693]},  # log1p(1) ≈ 0.693, i.e. ≥1 battle-related death
        # ── delta (multi-resolution spectral weight) ─────────────────────────────────
        # Spectral log_cosh(|S_pred| - |S_true|) at n_fft=6,12,24. DC bin masked.
        # Spectral operates on 36-step output vector only — does not compound through
        # hidden state unrolling. No BPTT tightening needed.
        "delta": {
            "distribution": "uniform",
            "min": 0.05,
            "max": 0.20,
        },
        # dual_mean=False: training loss = MSLE exactly. True caused false minimum.
        "dual_mean": {"values": [False]},
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config