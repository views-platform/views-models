def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    Optimized for TiDEModel on zero-inflated conflict fatalities data at country-month level.
    
    Returns:
    - sweep_config (dict): Configuration for hyperparameter sweeps.
    """

    sweep_config = {
        'method': 'bayes',
        'name': 'cool_cat_tide_balanced_v3',
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 10,
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
        'input_chunk_length': {'values': [36, 48]},  # Slightly shorter
        'output_chunk_shift': {'values': [0]},

        # ============== TRAINING BASICS ==============
        # batch_size: +0.83 importance → CRITICAL: MUCH smaller batches!
        # early_stopping_patience: -0.34 → higher patience helps
        # early_stopping_min_delta: +0.24 → smaller threshold needed
        'batch_size': {'values': [8, 16, 32, 24]},  # MUCH SMALLER (was 32-64)
        'n_epochs': {'values': [100]},
        'early_stopping_patience': {'values': [12]},  # HIGHER (was 18-25)
        'early_stopping_min_delta': {'values': [0.001]},
        'force_reset': {'values': [True]},

        # ============== OPTIMIZER / SCHEDULER ==============
        # lr: -0.53 importance → HIGHER LR is better for TiDE!
        # weight_decay: -0.4 → HIGHER weight_decay helps
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 5e-5,   # Higher (was 1e-5)
            'max': 2e-3,   # Higher (was 2e-4)
        },
        'weight_decay': {
            'distribution': 'uniform',
            'min': 5e-4,   # MUCH HIGHER (was 1e-5)
            'max': 5e-3,   # MUCH HIGHER (was 5e-4)
        },
        'lr_scheduler_factor': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.25,
        },
        'lr_scheduler_patience': {'values': [4]},  # Slightly higher
        'lr_scheduler_min_lr': {'values': [1e-7]},
        # gradient_clip_val: -0.076 → slightly higher helps
        'gradient_clip_val': {
            'distribution': 'uniform',
            'min': 0.01,
            'max': 1.2,  # Slightly higher range
        },

        # ============== SCALING ==============
        'feature_scaler': {'values': [None]},
        'target_scaler': {'values': ['AsinhTransform->MinMaxScaler']},  # Fixed best option
        'feature_scaler_map': {
            'values': [{
                # Zero-inflated conflict counts
                "AsinhTransform->MinMaxScaler": [
                    "lr_ged_sb", "lr_ged_ns", "lr_ged_os",
                    "lr_acled_sb", "lr_acled_sb_count", "lr_acled_os",
                    "lr_ged_sb_tsum_24",
                    "lr_splag_1_ged_sb", "lr_splag_1_ged_os", "lr_splag_1_ged_ns",
                    "lr_wdi_ny_gdp_mktp_kd", "lr_wdi_nv_agr_totl_kn",
                    "lr_wdi_sm_pop_netm", "lr_wdi_sm_pop_refg_or"
                ],
                "MinMaxScaler": [
                    "lr_wdi_sl_tlf_totl_fe_zs", "lr_wdi_se_enr_prim_fm_zs",
                    "lr_wdi_sp_urb_totl_in_zs", "lr_wdi_sh_sta_maln_zs", "lr_wdi_sh_sta_stnt_zs",
                    "lr_wdi_dt_oda_odat_pc_zs", "lr_wdi_ms_mil_xpnd_gd_zs",
                    "lr_vdem_v2x_horacc", "lr_vdem_v2xnp_client", "lr_vdem_v2x_veracc",
                    "lr_vdem_v2x_divparctrl", "lr_vdem_v2xpe_exlpol", "lr_vdem_v2x_diagacc",
                    "lr_vdem_v2xpe_exlgeo", "lr_vdem_v2xpe_exlgender", "lr_vdem_v2xpe_exlsocgr",
                    "lr_vdem_v2x_ex_party", "lr_vdem_v2x_genpp", "lr_vdem_v2xeg_eqdr",
                    "lr_vdem_v2xcl_prpty", "lr_vdem_v2xeg_eqprotec", "lr_vdem_v2x_ex_military",
                    "lr_vdem_v2xcl_dmove", "lr_vdem_v2x_clphy", "lr_vdem_v2x_hosabort",
                    "lr_vdem_v2xnp_regcorr",
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
                "StandardScaler->MinMaxScaler": ["lr_wdi_sp_pop_grow"],
                "SqrtTransform->MinMaxScaler": ["lr_wdi_sp_dyn_imrt_fe_in"],
                "RobustScaler->MinMaxScaler": ["lr_topic_tokens_t1", "lr_topic_tokens_t1_splag"]
            }]
        },

        # ============== TiDE ARCHITECTURE ==============
        # num_encoder_layers: +0.06 → slightly fewer is fine
        'num_encoder_layers': {'values': [1, 2, 4, 6]},  # Simplified
        'num_decoder_layers': {'values': [1, 2, 3, 4]},
        'decoder_output_dim': {'values': [16, 32, 48, 64]},
        'hidden_size': {'values': [8, 16, 32, 64, 128, 192, 256]},
        
        # temporal_width_future: -0.2 → larger values help
        # temporal_hidden_size_past: -0.09 → larger values help
        'temporal_width_past': {'values': [4, 6, 8]},
        'temporal_width_future': {'values': [6, 8, 10]},  # Larger (was 4-8)
        'temporal_hidden_size_past': {'values': [48, 64, 80]},  # Larger (was 32-64)
        'temporal_hidden_size_future': {'values': [32, 48, 64]},
        'temporal_decoder_hidden': {'values': [32, 64, 96, 128, 256]},
        
        # Regularization & normalization
        # dropout: +0.01 → near zero importance, keep moderate
        'use_layer_norm': {'values': [True, False]},
        'dropout': {'values': [0.25, 0.3, 0.35, 0.45]},  # Moderate (was 0.35-0.45)
        'use_static_covariates': {'values': [True, False]},
        'use_reversible_instance_norm': {'values': [False]},

        # ============== LOSS FUNCTION ==============
        # non_zero_weight: +0.4 → LOWER values are better!
        # delta: +0.4 → LOWER delta is better
        # false_negative_weight: +0.124 → slightly lower is better
        # false_positive_weight: +0.03 → near zero importance
        'loss_function': {'values': ['WeightedPenaltyHuberLoss']},
        
        'zero_threshold': {
            'distribution': 'log_uniform_values',
            'min': 0.001,
            'max': 0.1,
        },
        
        # delta: +0.4 importance → LOWER is better
        'delta': {
            'distribution': 'uniform',
            'min': 0.02,
            'max': 0.08,  # Much lower (was 0.1-0.4)
        },
        
        # non_zero_weight: +0.4 importance → LOWER is better
        'non_zero_weight': {
            'distribution': 'uniform',
            'min': 2.0,   # Lower (was 5.0)
            'max': 12.0,   # Lower (was 10.0)
        },
        
        # false_positive_weight: +0.03 → near zero, keep low-moderate
        'false_positive_weight': {
            'distribution': 'uniform',
            'min': 1.0,
            'max': 5.5,
        },
        
        # false_negative_weight: +0.124 → slightly lower is better
        'false_negative_weight': {
            'distribution': 'uniform',
            'min': 1.0,   # Lower (was 5.0)
            'max': 12.0,   # Lower (was 12.0)
        },
    }

    sweep_config['parameters'] = parameters
    return sweep_config