def get_sweep_config():
    """
    meow
    """
    sweep_config = {
        "method": "bayes",
        "name": "good_life_transformer_spotlight_lrop_20260503_round2",
        "early_terminate": {"type": "hyperband", "min_iter": 30, "eta": 2},  # Rungs at 30,90,270. min_iter=30 = 5 epochs post-restart-1, past the spike; comparisons at matched post-restart phase.
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
        "detect_anomaly": {"values": [False]},
        "optimizer_cls": {"values": ["AdamW"]},
        "num_samples": {"values": [1]},
        "n_jobs": {"values": [-1]},
        # ==============================================================================
        # TRAINING
        # ==============================================================================
        "batch_size": {"values": [128]},
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [50]},
        "early_stopping_min_delta": {"values": [0.0]},
        "force_reset": {"values": [True]},
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        "lr": {"values": [1e-3]},
        "weight_decay": {"values": [1e-4]},
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["ReduceLROnPlateau"]},
        "lr_scheduler_factor": {"values": [0.7]},
        "lr_scheduler_patience": {"values": [15]},
        "lr_scheduler_min_lr": {"values": [1e-6]},
        "lr_scheduler_kwargs": {"values": [{"mode": "min", "factor": 0.7, "patience": 15, "min_lr": 1e-6, "threshold": 0.50, "threshold_mode": "abs", "cooldown": 5}]},
        "gradient_clip_val": {"values": [1.5, 3.0, 5.0]},
        # ==============================================================================
        # SCALING
        # ==============================================================================
        "feature_scaler": {"values": [None]},
        "target_scaler": {"values": ["AsinhTransform"]},  # asinh(x): SpotlightLoss operates in asinh space. non_zero_threshold=0.88=asinh(1).
        # feature_scaler_map: all features consolidated under AsinhTransform.
        # asinh(x) ≈ x for x∈[0,1] (output range [0, 0.881], derivative ≥0.71),
        # so V-Dem indices are unharmed. Removes MinMaxScaler’s fitted min/max
        # (leakage risk) and simplifies the scaling pipeline.
        "feature_scaler_map": {
            "values": [{
                "AsinhTransform->StandardScaler": [
                    # Heavy-tailed: conflict counts, GDP, refugees, ODA
                    "lr_splag_1_ged_sb", "lr_splag_1_ged_ns", "lr_splag_1_ged_os",
                    "lr_ged_ns", "lr_ged_os",
                    "lr_ged_sb_delta", "lr_ged_ns_delta", "lr_ged_os_delta",
                    "lr_wdi_ny_gdp_mktp_kd", "lr_wdi_nv_agr_totl_kn",
                    "lr_wdi_sm_pop_refg_or", "lr_wdi_dt_oda_odat_pc_zs",
                    "lr_wdi_sp_pop_grow", "lr_wdi_sp_urb_totl_in_zs",
                    "lr_wdi_sm_pop_netm", "lr_acled_sb", "lr_acled_sb_count",
                    "lr_acled_os", 
                    
                    # Bounded [0,1] or near-bounded: V-Dem indices, WDI rates
                    "lr_vdem_v2x_horacc", "lr_vdem_v2x_veracc", "lr_vdem_v2x_diagacc",
                    "lr_vdem_v2xnp_client", "lr_vdem_v2xnp_regcorr",
                    "lr_vdem_v2xpe_exlpol", "lr_vdem_v2xpe_exlgeo",
                    "lr_vdem_v2xpe_exlgender", "lr_vdem_v2xpe_exlsocgr",
                    "lr_vdem_v2x_divparctrl", "lr_vdem_v2x_ex_party",
                    "lr_vdem_v2x_ex_military", "lr_vdem_v2x_genpp",
                    "lr_vdem_v2xeg_eqdr", "lr_vdem_v2xcl_prpty",
                    "lr_vdem_v2xeg_eqprotec", "lr_vdem_v2xcl_dmove",
                    "lr_vdem_v2x_clphy",
                    "lr_wdi_ms_mil_xpnd_gd_zs", "lr_wdi_sh_sta_stnt_zs",
                    "lr_wdi_sh_sta_maln_zs", "lr_wdi_sl_tlf_totl_fe_zs",
                    "lr_wdi_se_enr_prim_fm_zs", "lr_wdi_sp_dyn_imrt_fe_in",
                ],
            }]
        },
        # ==============================================================================
        # TRANSFORMER ARCHITECTURE
        # ==============================================================================
        # d_model=128 confirmed: broke the v26 event_ratio=0.58 plateau.
        # Run 3 (d_model=128, nhead=4, icl=48, ReLU): event_ratio=0.878 — best
        # calibration ever seen. d_model=256 generated excess capacity for 90%
        # zero-inflated data; 128 = 2.7× expansion from 48 features, sufficient.
        # d_model=128, nhead=4 → head_dim=32 ✓
        "d_model": {"values": [64, 128]},
        # nhead: locked at 4. With d_model=128, nhead=8 → head_dim=16 — too coarse.
        # Run 1 (nhead=8, icl=36): event_ratio=1.517 (52% overprediction). Each head
        # projects only 16 dims from 48 positions, creating noisy attention patterns
        # that amplify conflict signals rather than precisely attending to them.
        # nhead=4 → head_dim=32 gives sufficient resolution for lag-1 persistence,
        # lag-12 seasonality, and escalation onset patterns.
        "nhead": {"values": [2]},
        "num_encoder_layers": {"values": [2, 3]},
        "dim_feedforward": {"values": [256, 512, 1024]},
        "activation": {"values": ["GELU", "SwiGLU"]},
        "norm_type": {"values": ["LayerNorm", "RMSNorm"]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        "dropout": {"values": [0.15, 0.25, 0.35]},
        "use_reversible_instance_norm": {"values": [True]},
        # ==============================================================================
        # LOSS FUNCTION: SpotlightLossLogcosh
        # ==============================================================================
        "loss_function": {"values": ["SpotlightLoss"]},
        "non_zero_threshold": {"values": [0.88]}, 
        # delta: multi-resolution spectral weight. DC bin masked.
        "delta": {"values": [0.075]},
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config