def get_sweep_config():
    """BlockRNN"""
    sweep_config = {
        "method": "bayes",
        "name": "dancing_queen_spotlight_v1_5",
        "early_terminate": {"type": "hyperband", "min_iter": 25, "eta": 2},  # > T_0=25: survives first CAWR spike + 5 recovery epochs before Hyperband kills
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        # 36 for BPTT safety, 48 for full context (Bayes decides).
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
        "batch_size": {"values": [128]},
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [30]},  # T_mult=1: constant 25-epoch cycles; patience=30 > T_0=25 so early stopping survives first CAWR reset
        "early_stopping_min_delta": {"values": [0.0001]},
        "force_reset": {"values": [True]},
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        # lr floor raised to 2e-4: runs below 2e-4 show systematic mean-regression underprediction
        # (model learns sparsity pattern but can't escape toward heavy tail in 100 epochs).
        "lr": {
            "distribution": "log_uniform_values",
            "min": 5e-5,
            "max": 1e-3,
        },
        # weight_decay fixed at 1e-4: 1e-2 and 1e-3 crush FC decoder output weights needed
        # for high-magnitude conflict predictions (weight_max 37→18 w/ 10x less decay).
        "weight_decay": {"values": [1e-4, 1e-3, 0]},
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["CosineAnnealingWarmRestarts"]},
        # T_0=25 causes 3-4 restart spikes before early stopping kills the run (loss_stability/cv=0.63).
        # T_0=50 doubles the descent window per cycle, loss stabilizes before patience triggers.
        "lr_scheduler_T_0": {"values": [25]},
        "lr_scheduler_T_mult": {"values": [1]},  # T_mult=1: cycles remain the same (25→25→25 epochs) — each restart the model gets the same time to settle before next LR spike
        "lr_scheduler_eta_min": {"values": [1e-6]},
        # clip_val=1.0 was hitting max grad_norm on every run (0.97-0.98 observed).
        # Capping learning on conflict spikes. 0.5 stabilizes; 2.0 lets strong gradients through.
        "gradient_clip_val": {"values": [1.0, 2.0, 3.0]},
        # ==============================================================================
        # SCALING
        # ==============================================================================
        "feature_scaler": {"values": [None]},
        "target_scaler": {"values": ["AsinhTransform"]},  # asinh(x): SpotlightLoss operates in asinh space. non_zero_threshold=0.88=asinh(1).
        "feature_scaler_map": {
            "values": [
                {
                    "AsinhTransform": [
                        "lr_wdi_sp_dyn_imrt_fe_in",
                        "lr_wdi_sm_pop_refg_or",
                        "lr_wdi_ny_gdp_mktp_kd",
                        "lr_wdi_nv_agr_totl_kn",
                        "lr_splag_1_ged_sb",
                        "lr_splag_1_ged_ns",
                        "lr_splag_1_ged_os",
                        "lr_ged_ns", 
                        "lr_ged_os",
                        "lr_ged_sb_delta",
                        "lr_ged_ns_delta",
                        "lr_ged_os_delta",
                        "lr_wdi_sm_pop_netm",
                        "lr_wdi_dt_oda_odat_pc_zs",
                        "lr_wdi_sp_pop_grow",
                        "lr_wdi_ms_mil_xpnd_gd_zs",
                        "lr_wdi_sh_sta_stnt_zs",
                        "lr_wdi_sh_sta_maln_zs",
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
        # n_rnn_layers=1 is a confirmed architectural ceiling: both 128 and 256 hidden runs
        # plateau at MSLE_mean=0.532 with identical scores. Layer 2 encodes trend/regime on top of
        # layer 1's local dynamics — critical for multi-scale conflict signal.
        "n_rnn_layers": {"values": [1, 2]},
        # hidden_fc_sizes: [256] for hidden=256 (1:1). [128] for hidden=128 (1:1). [256,128] narrows decoder.
        "hidden_fc_sizes": {"values": [[128], [256], [256, 128]]},
        # GELU: smoother gradients through the FC decoder.
        "activation": {"values": ["GELU"]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        "dropout": {"values": [0.15, 0.25]},
        "use_static_covariates": {"values": [True]},
        # RevIN off: peace series have σ≈0 in asinh space → RevIN divides by near-zero σ → instability.
        # Also conflicts with GRU hidden-state trend tracking (RevIN normalizes input window + denorms output).
        "use_reversible_instance_norm": {"values": [True]},
        "static_covariate_stats": {"values": [{"transform": "AsinhTransform"}]},
        "loss_function": {"values": ["SpotlightLoss"]},
        "non_zero_threshold": {"values": [0.88]}, 
        # delta: multi-resolution spectral weight. DC bin masked.
        # Ceiling lowered to 0.10: high delta (>0.12) shifts loss toward spectral shape over amplitude,
        # compounding underprediction when combined with sparse conflict data.
        "delta": {"distribution": "uniform", "min": 0.05, "max": 0.10},
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config