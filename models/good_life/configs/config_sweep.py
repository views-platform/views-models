def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    Optimized for TransformerModel on zero-inflated conflict fatalities data.
    
    Transformer Architecture Notes:
    - Self-attention captures long-range temporal dependencies
    - Multi-head attention allows learning different temporal patterns
    - d_model must be divisible by nhead
    - Deeper models (more layers) may overfit on sparse data
    - Lower learning rates critical for training stability
    
    Returns:
    - sweep_config (dict): Configuration for hyperparameter sweeps.
    """

    sweep_config = {
        'method': 'bayes',
        'name': 'good_life_transformer_chained_scalers_cm',
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
        'steps': {'values': [[*range(1, 36 + 1)]]},
        'input_chunk_length': {'values': [36, 48, 60]},
        'output_chunk_length': {'values': [36]},

        # ============== TRAINING BASICS ==============
        'batch_size': {'values': [32, 64, 128, 256]},
        'n_epochs': {'values': [300]},
        'early_stopping_patience': {'values': [15, 25]},  # Transformers need patience
        'early_stopping_min_delta': {'values': [0.001, 0.005]},
        'force_reset': {'values': [True]},

        # ============== OPTIMIZER / SCHEDULER ==============
        # Transformers need lower learning rates for stability
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 2e-4,
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-3,
        },
        'lr_scheduler_factor': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.4,
        },
        'lr_scheduler_patience': {'values': [3, 5, 7]},
        'lr_scheduler_min_lr': {'values': [1e-6]},
        'gradient_clip_val': {
            'distribution': 'uniform',
            'min': 0.5,
            'max': 1.0,
        },

        # ============== SCALING ==============
        # RobustScaler as default fallback for unmapped features
        # feature_scaler_map assigns optimal scalers based on feature characteristics:
        # - AsinhTransform: Zero-inflated counts (fatalities) and heavily skewed economic data
        # - MinMaxScaler: Bounded percentages (0-100), V-Dem indices (0-1), topic proportions
        # - StandardScaler: Growth rates (normal-ish, can be negative)
        # - SqrtTransform: Mortality rates (positive, moderate skew)
        # NOTE: Do NOT use log_targets=True with AsinhTransform - causes double transform and NaN loss!
        'log_targets': {'values': [False]},
        'feature_scaler': {'values': [None]},
        'target_scaler': {'values': ['AsinhTransform->MinMaxScaler', 'RobustScaler->MinMaxScaler']},
        'feature_scaler_map': {
            'values': [{
                # Zero-inflated conflict counts - asinh handles zeros and extreme spikes
                "AsinhTransform->MinMaxScaler": [
                    "ged_sb", "ged_sb_dep", "ged_ns", "ged_os",
                    "acled_sb", "acled_sb_count", "acled_os",
                    "ged_sb_tsum_24",
                    "splag_1_ged_sb", "splag_1_ged_os", "splag_1_ged_ns",
                    # Large-scale economic data with extreme skew
                    "wdi_ny_gdp_mktp_kd", "wdi_nv_agr_totl_kn",
                    "wdi_sm_pop_netm", "wdi_sm_pop_refg_or"
                ],
                # Bounded percentages and rates (0-100 scale)
                "MinMaxScaler": [
                    "wdi_sl_tlf_totl_fe_zs", "wdi_se_enr_prim_fm_zs",
                    "wdi_sp_urb_totl_in_zs", "wdi_sh_sta_maln_zs", "wdi_sh_sta_stnt_zs",
                    "wdi_dt_oda_odat_pc_zs", "wdi_ms_mil_xpnd_gd_zs",
                    # V-Dem indices (already 0-1 bounded)
                    "vdem_v2x_horacc", "vdem_v2xnp_client", "vdem_v2x_veracc",
                    "vdem_v2x_divparctrl", "vdem_v2xpe_exlpol", "vdem_v2x_diagacc",
                    "vdem_v2xpe_exlgeo", "vdem_v2xpe_exlgender", "vdem_v2xpe_exlsocgr",
                    "vdem_v2x_ex_party", "vdem_v2x_genpp", "vdem_v2xeg_eqdr",
                    "vdem_v2xcl_prpty", "vdem_v2xeg_eqprotec", "vdem_v2x_ex_military",
                    "vdem_v2xcl_dmove", "vdem_v2x_clphy", "vdem_v2x_hosabort",
                    "vdem_v2xnp_regcorr",
                    # Topic model proportions (0-1 bounded)
                    "topic_ste_theta0", "topic_ste_theta1", "topic_ste_theta2",
                    "topic_ste_theta3", "topic_ste_theta4", "topic_ste_theta5",
                    "topic_ste_theta6", "topic_ste_theta7", "topic_ste_theta8",
                    "topic_ste_theta9", "topic_ste_theta10", "topic_ste_theta11",
                    "topic_ste_theta12", "topic_ste_theta13", "topic_ste_theta14",
                    "topic_ste_theta0_stock_t1_splag", "topic_ste_theta1_stock_t1_splag",
                    "topic_ste_theta2_stock_t1_splag", "topic_ste_theta3_stock_t1_splag",
                    "topic_ste_theta4_stock_t1_splag", "topic_ste_theta5_stock_t1_splag",
                    "topic_ste_theta6_stock_t1_splag", "topic_ste_theta7_stock_t1_splag",
                    "topic_ste_theta8_stock_t1_splag", "topic_ste_theta9_stock_t1_splag",
                    "topic_ste_theta10_stock_t1_splag", "topic_ste_theta11_stock_t1_splag",
                    "topic_ste_theta12_stock_t1_splag", "topic_ste_theta13_stock_t1_splag",
                    "topic_ste_theta14_stock_t1_splag"
                ],
                # Growth rates (can be negative, roughly normal)
                "StandardScaler->MinMaxScaler": [
                    "wdi_sp_pop_grow"
                ],
                # Mortality rates (positive, moderate skew)
                "SqrtTransform->MinMaxScaler": [
                    "wdi_sp_dyn_imrt_fe_in"
                ],
                # Token counts (moderate skew)
                "RobustScaler->MinMaxScaler": [
                    "topic_tokens_t1", "topic_tokens_t1_splag"
                ]
            }]
        },

        # ============== TRANSFORMER ARCHITECTURE ==============
        # Note: d_model must be divisible by nhead
        # Constraint: d_model / nhead >= 16 for stable attention (avoid 64/8=8 which can cause NaN)
        'd_model': {'values': [128, 256]},  # Larger embedding dims more stable
        'num_attention_heads': {'values': [4, 8]},  # 128/4=32, 128/8=16, 256/4=64, 256/8=32 - all safe
        'num_encoder_layers': {'values': [2, 3]},  # Moderate depth, less prone to vanishing gradients
        'num_decoder_layers': {'values': [2, 3]},
        'dim_feedforward': {'values': [256, 512, 1024]},  # FFN dimension
        'dropout': {'values': [0.2, 0.3]},  # Moderate dropout
        'activation': {'values': ['gelu']},  # GELU more stable than ReLU for transformers
        'norm_type': {'values': ['LayerNorm']},  # LayerNorm standard for transformers
        'use_reversible_instance_norm': {'values': [True]},  # Helps with non-stationarity

        # ============== LOSS FUNCTION ==============
        'loss_function': {'values': ['WeightedPenaltyHuberLoss']},
        
        'zero_threshold': {
            'distribution': 'uniform',
            'min': 0.01,
            'max': 0.1,
        },
        'delta': {
            'distribution': 'log_uniform_values',
            'min': 0.1,
            'max': 0.8,
        },
        'non_zero_weight': {
            'distribution': 'uniform',
            'min': 4.0,
            'max': 8.0,
        },
        'false_positive_weight': {
            'distribution': 'uniform',
            'min': 1.5,
            'max': 3.0,
        },
        'false_negative_weight': {
            'distribution': 'uniform',
            'min': 2.5,
            'max': 6.0,
        },
    }

    sweep_config['parameters'] = parameters
    return sweep_config