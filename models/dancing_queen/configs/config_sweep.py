def get_sweep_config():
    """
    Key changes from v4:
    - SpotlightLoss v20: removed dead params (beta, kappa, gamma).
    - Tighter alpha/delta for BPTT safety: alpha [0.15,0.25], delta [0.10,0.22].
    - Locked: GRU, n_rnn_layers=1, hidden_fc_sizes=[128], GELU, dropout=0.0.
    - RevIN=True for zero-inflated input normalization.
    - weight_decay=1e-4, gradient_clip_val=2.0, min_iter=50, patience=50.
    - icl [36,48]: 36 for BPTT safety, 48 for full context.
    - hidden_dim [64,128]: 64 is sufficient for ~200 series.
    - Added lr_ged_ns/os to feature_scaler_map.
    - use_cyclic_encoders=True replaces positional encoders.
    """
    sweep_config = {
        "method": "bayes",
        "name": "dancing_queen_blockrnn_spotlight_v5_msle",
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
            "min": 1e-5,
            "max": 5e-4,
        },
        "weight_decay": {"values": [1e-4]},
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["CosineAnnealingWarmRestarts"]},
        "lr_scheduler_T_0": {"values": [30]},
        "lr_scheduler_T_mult": {"values": [2]},
        "lr_scheduler_eta_min": {"values": [1e-6]},
        "gradient_clip_val": {"values": [2.0]},
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
        # BLOCKRNN ARCHITECTURE
        # ==============================================================================
        # GRU locked: update gate handles zero-heavy sequences better than
        # LSTM's forget gate (which aggressively zeros cell state).
        "rnn_type": {"values": ["GRU"]},
        # hidden_dim: 64-128 for ~200 series. Single layer keeps all params
        # reachable by gradient.
        "hidden_dim": {"values": [64, 128]},
        # Locked to 1: removes inter-layer dropout dead zone, halves BPTT
        # memory, keeps all hidden state reachable by gradient.
        "n_rnn_layers": {"values": [1]},
        # Single 128-wide layer: 1.2 units per output (108 = 36×3).
        # Enough for target-specific discrimination without OOD capacity.
        "hidden_fc_sizes": {"values": [[128]]},
        # GELU: smoother gradients through the FC decoder.
        "activation": {"values": ["GELU"]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        "dropout": {"values": [0.10, 0.15, 0.25]},
        "use_static_covariates": {"values": [True]},
        "use_reversible_instance_norm": {"values": [True]},
        # ==============================================================================
        # LOSS FUNCTION: SpotlightLoss v20
        # ==============================================================================
        "loss_function": {"values": ["SpotlightLoss"]},
        # ── alpha (magnitude weighting) ───────────────
        # Tighter for BPTT: gradients compound across time steps.
        # [0.15, 0.25] keeps cosh(0.25*11.5) ≈ 5.4x at max asinh value.
        "alpha": {
            "distribution": "uniform",
            "min": 0.15,
            "max": 0.25,
        },
        # ── delta (spectral resolution) ───────────────
        # Tighter for BPTT: STFT gradients compound through unrolled steps.
        "delta": {
            "distribution": "uniform",
            "min": 0.10,
            "max": 0.22,
        },
        # ── non_zero_threshold ────────────────────────
        "non_zero_threshold": {"values": [0.88]},
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config