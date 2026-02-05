def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    Optimized for BlockRNNModel (LSTM/GRU) on zero-inflated conflict fatalities data.
    
    BlockRNN Architecture Notes:
    - Uses stacked LSTM/GRU blocks for sequential pattern learning
    - GRU often faster and comparable performance to LSTM
    - Hidden state captures temporal dependencies in conflict escalation
    
    Returns:
    - sweep_config (dict): Configuration for hyperparameter sweeps.
    """

    sweep_config = {
        'method': 'bayes',
        'name': 'dancing_queen_blockrnn_balanced_v2_mtd',
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 20,
            'eta': 2
        },
        'metric': {
            'name': 'time_series_wise_mtd_mean_sb',
            'goal': 'minimize'
        },
    }

    parameters = {
        # ============== TEMPORAL CONFIGURATION ==============
        # input_chunk_length: -0.12 → LONGER context helps
        'steps': {'values': [[*range(1, 36 + 1)]]},
        'input_chunk_length': {'values': [36, 48]},
        'output_chunk_shift': {'values': [0]},

        # ============== TRAINING BASICS ==============
        # batch_size: +0.08 → slightly smaller helps
        # early_stopping_patience: +0.1 → slightly lower
        # early_stopping_min_delta: +0.04 → smaller threshold
        'batch_size': {'values': [512, 1024, 2048, 4096]},
        'n_epochs': {'values': [200]},
        'early_stopping_patience': {'values': [15, 20, 25]},
        "early_stopping_min_delta": {"values": [0.00005, 0.0001]},  # Smaller for [0,1] loss scale
        'force_reset': {'values': [True]},

        # ============== OPTIMIZER / SCHEDULER ==============
        # lr: -0.3 → HIGHER LR is better!
        # lr_scheduler_factor: -0.2 → more aggressive decay
        # gradient_clip_val: -0.13 → HIGHER clipping helps
        # weight_decay: +0.11 → LOWER is better
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 5e-5,
            'max': 1e-3,
        },
        'weight_decay': {'values': [0]},
        'lr_scheduler_factor': {'values': [0.5]},  # Fixed for stability
        'lr_scheduler_patience': {'values': [8]},  # Fixed - consistent plateau detection
        'lr_scheduler_min_lr': {'values': [1e-6]},  # Higher floor
        "gradient_clip_val": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1.5,
        },

        # ============== SCALING ==============
        # RobustScaler as default fallback for unmapped features
        # feature_scaler_map assigns optimal scalers based on feature characteristics:
        # - AsinhTransform: Zero-inflated counts (fatalities) and heavily skewed economic data
        # - MinMaxScaler: Bounded percentages (0-100), V-Dem indices (0-1), topic proportions
        # - StandardScaler: Growth rates (normal-ish, can be negative)
        # - SqrtTransform: Mortality rates (positive, moderate skew)
        'feature_scaler': {'values': [None]},
        # Target is lr_ged_sb - use same scaling as the feature for consistency
        'target_scaler': {'values': ['AsinhTransform->MinMaxScaler']},
        'feature_scaler_map': {
            'values': [{
                # Zero-inflated conflict counts - Asinh + StandardScaler preserves gradients
                "AsinhTransform->MinMaxScaler": [
                    "lr_ged_sb", "lr_ged_ns", "lr_ged_os",
                    "lr_acled_sb", "lr_acled_sb_count", "lr_acled_os",
                    "lr_ged_sb_tsum_24",
                    "lr_splag_1_ged_sb", "lr_splag_1_ged_os", "lr_splag_1_ged_ns",
                    # Large-scale economic data with extreme skew
                    "lr_wdi_ny_gdp_mktp_kd", "lr_wdi_nv_agr_totl_kn",
                    "lr_wdi_sm_pop_netm", "lr_wdi_sm_pop_refg_or",
                ],
                # Bounded percentages, V-Dem indices, and growth rates - StandardScaler works fine
                "MinMaxScaler": [
                    "lr_wdi_sl_tlf_totl_fe_zs", "lr_wdi_se_enr_prim_fm_zs",
                    "lr_wdi_sp_urb_totl_in_zs", "lr_wdi_sh_sta_maln_zs", "lr_wdi_sh_sta_stnt_zs",
                    "lr_wdi_dt_oda_odat_pc_zs", "lr_wdi_ms_mil_xpnd_gd_zs",
                    # V-Dem indices (already 0-1 bounded)
                    "lr_vdem_v2x_horacc", "lr_vdem_v2xnp_client", "lr_vdem_v2x_veracc",
                    "lr_vdem_v2x_divparctrl", "lr_vdem_v2xpe_exlpol", "lr_vdem_v2x_diagacc",
                    "lr_vdem_v2xpe_exlgeo", "lr_vdem_v2xpe_exlgender", "lr_vdem_v2xpe_exlsocgr",
                    "lr_vdem_v2x_ex_party", "lr_vdem_v2x_genpp", "lr_vdem_v2xeg_eqdr",
                    "lr_vdem_v2xcl_prpty", "lr_vdem_v2xeg_eqprotec", "lr_vdem_v2x_ex_military",
                    "lr_vdem_v2xcl_dmove", "lr_vdem_v2x_clphy", "lr_vdem_v2x_hosabort",
                    "lr_vdem_v2xnp_regcorr",
                    # Topic model proportions (0-1 bounded)
                    "lr_topic_ste_theta0", "lr_topic_ste_theta1", "lr_topic_ste_theta2",
                    "lr_topic_ste_theta3", "lr_topic_ste_theta4", "lr_topic_ste_theta5",
                    "lr_topic_ste_theta6", "lr_topic_ste_theta7", "lr_topic_ste_theta8",
                    "lr_topic_ste_theta9", "lr_topic_ste_theta10", "lr_topic_ste_theta11",
                    "lr_topic_ste_theta12", "lr_topic_ste_theta13", "lr_topic_ste_theta14",
                    "lr_topic_ste_theta0_stock_t1_splag", "lr_topic_ste_theta1_stock_t1_splag",
                    "lr_topic_ste_theta2_stock_t1_splag", "lr_topic_ste_theta3_stock_t1_splag",
                    "lr_topic_ste_theta4_stock_t1_splag", "lr_topic_ste_theta5_stock_t1_splag",
                    "lr_topic_ste_theta6_stock_t1_splag", "lr_topic_ste_theta7_stock_t1_splag",
                    "lr_topic_ste_theta8_stock_t1_splag", "lr_topic_ste_theta9_stock_t1_splag",
                    "lr_topic_ste_theta10_stock_t1_splag", "lr_topic_ste_theta11_stock_t1_splag",
                    "lr_topic_ste_theta12_stock_t1_splag", "lr_topic_ste_theta13_stock_t1_splag",
                    "lr_topic_ste_theta14_stock_t1_splag",
                    "lr_topic_tokens_t1", "lr_topic_tokens_t1_splag",
                    # Growth rates (can be negative, roughly normal)
                    "lr_wdi_sp_pop_grow",
                    # Mortality rates (positive, skewed)
                    "lr_wdi_sp_dyn_imrt_fe_in",
                ],
            }]
        },

        # ============== BLOCKRNN ARCHITECTURE ==============
        # hidden_dim: +0.32 → SMALLER is better
        # dropout: +0.4 → LOWER is better
        # n_rnn_layers: +0.02 → near zero importance
        'rnn_type': {'values': ['LSTM', 'GRU']},
        'hidden_dim': {'values': [16, 32, 64, 128, 192]},  # SMALLER (was 128-512)
        'n_rnn_layers': {'values': [1, 2, 3, 4]},  # Simplified (was 2-4)
        'activation': {'values': ['ReLU', 'GELU']},  # Removed Tanh
        'dropout': {'values': [0.05, 0.1, 0.15]},  # LOW dropout - preserve neurons that learn rare patterns
        'use_reversible_instance_norm': {'values': [False, True]},

        # ============== LOSS FUNCTION ==============
        # delta: +0.5 → LOWER delta (more L1-like, anti-smoothing!)
        # false_negative_weight: +0.4 → LOWER is better
        # non_zero_weight: +0.2 → LOWER is better
        # zero_threshold: +0.2 → LOWER is better
        'loss_function': {'values': ['WeightedPenaltyHuberLoss']},
        
        # Zero threshold - what counts as "zero" after scaling
        'zero_threshold': {
            'distribution': 'uniform',  # uniform is fine for this small range
            'min': 0.05,   # Safely above 0 noise
            'max': 0.18,   # Just above where 1 fatality lands (~0.11)
        },
        # Delta for Huber loss - tighter range for consistent gradient flow
        'delta': {
            'distribution': 'uniform',
            'min': 0.8,
            'max': 1.0,  # Full L2
        },
        
        # Non-zero weight - narrower range for stability
        'non_zero_weight': {
            'distribution': 'uniform',
            'min': 4.0,
            'max': 7.0,  # Narrower range prevents conflicting gradients
        },
        
        "false_positive_weight": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1.0,
        },
        
        # False negative weight - narrower range
        'false_negative_weight': {
            'distribution': 'uniform',
            'min': 2.0,
            'max': 5.0,  # Narrower - still emphasizes missing conflicts
        },
    }

    sweep_config['parameters'] = parameters
    return sweep_config