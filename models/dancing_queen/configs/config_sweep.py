def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    Optimized for BlockRNNModel (LSTM/GRU) on zero-inflated conflict fatalities data.
    
    BlockRNN Architecture Notes:
    - Uses stacked LSTM/GRU blocks for sequential pattern learning
    - GRU often faster and comparable performance to LSTM
    - Hidden state captures temporal dependencies in conflict escalation
    
    Parameter Importance Analysis:
    - delta: +0.5 → LOWER delta (more L1-like, anti-smoothing)
    - dropout: +0.4 → LOWER dropout is better
    - false_negative_weight: +0.4 → LOWER FN weight is better
    - hidden_dim: +0.32 → SMALLER hidden_dim
    - lr: -0.3 → HIGHER LR is better
    - lr_scheduler_factor: -0.2 → more aggressive decay helps
    - gradient_clip_val: -0.13 → HIGHER clipping helps
    - input_chunk_length: -0.12 → LONGER context helps
    
    Anti-Smoothing Strategy:
    - Very low delta (0.05-0.15) for L1-like loss
    - Higher LR with aggressive decay
    - Smaller hidden_dim to prevent over-capacity
    - Lower dropout (model is regularized via loss function)
    
    Returns:
    - sweep_config (dict): Configuration for hyperparameter sweeps.
    """

    sweep_config = {
        'method': 'bayes',
        'name': 'dancing_queen_blockrnn_balanced_cd_v1',
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 12,
            'eta': 2
        },
        'metric': {
            'name': 'time_series_wise_msle_mean_sb',
            'goal': 'minimize'
        },
    }

    parameters = {
        # ============== TEMPORAL CONFIGURATION ==============
        # input_chunk_length: -0.12 → LONGER context helps
        'steps': {'values': [[*range(1, 36 + 1)]]},
        'input_chunk_length': {'values': [36, 48, 60]},  # Longer (was 24-48)
        'output_chunk_shift': {'values': [0]},

        # ============== TRAINING BASICS ==============
        # batch_size: +0.08 → slightly smaller helps
        # early_stopping_patience: +0.1 → slightly lower
        # early_stopping_min_delta: +0.04 → smaller threshold
        'batch_size': {'values': [64, 128]},  # Slightly smaller (was 64-256)
        'n_epochs': {'values': [350]},
        'early_stopping_patience': {'values': [8, 10, 12]},  # Slightly lower
        'early_stopping_min_delta': {'values': [0.0005, 0.001]},  # Smaller
        'force_reset': {'values': [True]},

        # ============== OPTIMIZER / SCHEDULER ==============
        # lr: -0.3 → HIGHER LR is better!
        # lr_scheduler_factor: -0.2 → more aggressive decay
        # gradient_clip_val: -0.13 → HIGHER clipping helps
        # weight_decay: +0.11 → LOWER is better
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 2e-4,   # Higher (was 5e-5)
            'max': 1e-3,   # Higher (was 5e-4)
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,   # Lower (was 1e-5)
            'max': 1e-4,   # Lower (was 1e-3)
        },
        'lr_scheduler_factor': {
            'distribution': 'uniform',
            'min': 0.05,   # More aggressive (was 0.1)
            'max': 0.2,    # More aggressive (was 0.4)
        },
        'lr_scheduler_patience': {'values': [3, 4, 5]},
        'lr_scheduler_min_lr': {'values': [1e-6]},
        'gradient_clip_val': {
            'distribution': 'uniform',
            'min': 0.5,    # Higher (was 0.1)
            'max': 1.5,    # Higher (was 1.0)
        },

        # ============== SCALING ==============
        # RobustScaler as default fallback for unmapped features
        # feature_scaler_map assigns optimal scalers based on feature characteristics:
        # - AsinhTransform: Zero-inflated counts (fatalities) and heavily skewed economic data
        # - MinMaxScaler: Bounded percentages (0-100), V-Dem indices (0-1), topic proportions
        # - StandardScaler: Growth rates (normal-ish, can be negative)
        # - SqrtTransform: Mortality rates (positive, moderate skew)
        'feature_scaler': {'values': [None]},
        'target_scaler': {'values': ['AsinhTransform->MinMaxScaler']},  # Fixed best option
        'feature_scaler_map': {
            'values': [{
                # Zero-inflated conflict counts - asinh handles zeros and extreme spikes
                "AsinhTransform->MinMaxScaler": [
                    "lr_ged_sb", "lr_ged_ns", "lr_ged_os",
                    "lr_acled_sb", "lr_acled_sb_count", "lr_acled_os",
                    "lr_ged_sb_tsum_24",
                    "lr_splag_1_ged_sb", "lr_splag_1_ged_os", "lr_splag_1_ged_ns",
                    # Large-scale economic data with extreme skew
                    "lr_wdi_ny_gdp_mktp_kd", "lr_wdi_nv_agr_totl_kn",
                    "lr_wdi_sm_pop_netm", "lr_wdi_sm_pop_refg_or"
                ],
                # Bounded percentages and rates (0-100 scale)
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
                    "lr_topic_ste_theta14_stock_t1_splag"
                ],
                # Growth rates (can be negative, roughly normal)
                "StandardScaler->MinMaxScaler": [
                    "lr_wdi_sp_pop_grow"
                ],
                # Mortality rates (positive, moderate skew)
                "SqrtTransform->MinMaxScaler": [
                    "lr_wdi_sp_dyn_imrt_fe_in"
                ],
                # Token counts (moderate skew)
                "RobustScaler->MinMaxScaler": [
                    "lr_topic_tokens_t1", "lr_topic_tokens_t1_splag"
                ]
            }]
        },

        # ============== BLOCKRNN ARCHITECTURE ==============
        # hidden_dim: +0.32 → SMALLER is better
        # dropout: +0.4 → LOWER is better
        # n_rnn_layers: +0.02 → near zero importance
        'rnn_type': {'values': ['LSTM', 'GRU']},
        'hidden_dim': {'values': [64, 128, 192]},  # SMALLER (was 128-512)
        'n_rnn_layers': {'values': [2, 3]},  # Simplified (was 2-4)
        'activation': {'values': ['ReLU', 'GELU']},  # Removed Tanh
        'dropout': {'values': [0.1, 0.15, 0.2]},  # LOWER (was 0.2-0.4)
        'use_reversible_instance_norm': {'values': [False]},

        # ============== LOSS FUNCTION ==============
        # delta: +0.5 → LOWER delta (more L1-like, anti-smoothing!)
        # false_negative_weight: +0.4 → LOWER is better
        # non_zero_weight: +0.2 → LOWER is better
        # zero_threshold: +0.2 → LOWER is better
        'loss_function': {'values': ['WeightedPenaltyHuberLoss']},
        
        # zero_threshold: +0.2 → LOWER is better
        'zero_threshold': {'values': [0.005, 0.01]},  # Lower (was 0.01-0.1)
        
        # delta: +0.5 → MUCH LOWER (key for anti-smoothing!)
        'delta': {
            'distribution': 'log_uniform_values',
            'min': 0.05,
            'max': 0.15,  # MUCH lower (was 0.1-0.8)
        },
        
        # non_zero_weight: +0.2 → LOWER is better
        'non_zero_weight': {
            'distribution': 'uniform',
            'min': 2.0,   # Lower (was 4.0)
            'max': 5.0,   # Lower (was 8.0)
        },
        
        # false_positive_weight: near zero importance, keep moderate
        'false_positive_weight': {
            'distribution': 'uniform',
            'min': 1.5,
            'max': 3.0,
        },
        
        # false_negative_weight: +0.4 → LOWER is better
        'false_negative_weight': {
            'distribution': 'uniform',
            'min': 1.5,   # Lower (was 2.5)
            'max': 4.0,   # Lower (was 6.0)
        },
    }

    sweep_config['parameters'] = parameters
    return sweep_config