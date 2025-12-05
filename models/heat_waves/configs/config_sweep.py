def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    Optimized for TFTModel (Temporal Fusion Transformer) on zero-inflated conflict fatalities data.
    
    TFT Architecture Notes:
    - Combines LSTM temporal processing with multi-head attention
    - Variable selection networks identify important features
    - Gated residual connections allow learning complex patterns
    - Interpretable attention weights show temporal importance
    - Most complex Darts model - needs careful regularization
    
    Returns:
    - sweep_config (dict): Configuration for hyperparameter sweeps.
    """

    sweep_config = {
        'method': 'bayes',
        'name': 'heat_waves_tft_sweep_week_dylan',
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
        'batch_size': {'values': [64, 128, 256]},
        'n_epochs': {'values': [300]},
        'early_stopping_patience': {'values': [12, 15]},
        'early_stopping_min_delta': {'values': [0.001, 0.005]},
        'force_reset': {'values': [True]},

        # ============== OPTIMIZER / SCHEDULER ==============
        # TFT is complex - needs lower learning rates
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
        'feature_scaler': {'values': [None]},
        'target_scaler': {'values': ['AsinhTransform', 'RobustScaler']},
        'feature_scaler_map': {
            'values': [{
                # Zero-inflated conflict counts - asinh handles zeros and extreme spikes
                "AsinhTransform": [
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
                "StandardScaler": [
                    "wdi_sp_pop_grow"
                ],
                # Mortality rates (positive, moderate skew)
                "SqrtTransform": [
                    "wdi_sp_dyn_imrt_fe_in"
                ],
                # Token counts (moderate skew)
                "RobustScaler": [
                    "topic_tokens_t1", "topic_tokens_t1_splag"
                ]
            }]
        },

        # ============== TFT ARCHITECTURE ==============
        'hidden_size': {'values': [64, 128, 256]},  # Main hidden dimension
        'lstm_layers': {'values': [1, 2, 3]},  # LSTM encoder layers
        'num_attention_heads': {'values': [4, 8]},  # Multi-head attention
        'dropout': {'values': [0.2, 0.3, 0.4]},  # Higher dropout for complex model
        'full_attention': {'values': [True]},  # Full attention for long sequences
        'feed_forward': {'values': ['GatedResidualNetwork', 'GLU', 'ReGLU', 'GEGLU']},  # Best variants
        'add_relative_index': {'values': [True]},  # Temporal position encoding
        'use_static_covariates': {'values': [True]},  # Enable for country features
        'norm_type': {'values': ['LayerNorm']},
        'use_reversible_instance_norm': {'values': [True]},  # Helps with distribution shift

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