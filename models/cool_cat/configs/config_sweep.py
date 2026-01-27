def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    Optimized for TiDEModel on zero-inflated conflict fatalities data at country-month level.
    
    TiDE (Time-series Dense Encoder) Architecture Notes:
    - Uses MLPs for encoding past and future covariates
    - Temporal projections compress time dimension before dense layers
    - Layer normalization critical for stability with sparse data
    - Reversible instance normalization helps with non-stationary conflict patterns
    
    Returns:
    - sweep_config (dict): Configuration for hyperparameter sweeps.
    """

    sweep_config = {
        'method': 'bayes',
        'name': 'cool_cat_tide_cm_aggressive',
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 20,  # Increased: let runs go longer before terminating
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
        'input_chunk_length': {'values': [48, 60, 72]},  # Longer context helps avoid smoothing
        'output_chunk_shift': {'values': [0]},

        # ============== TRAINING BASICS ==============
        # Higher patience strongly correlated with lower MSLE (-0.7)
        'batch_size': {'values': [32, 64]},  # Smaller batches = more gradient updates
        'n_epochs': {'values': [400]},  # More epochs since we want longer training
        'early_stopping_patience': {'values': [18, 22, 25]},  # MUCH HIGHER (was 10-15)
        'early_stopping_min_delta': {'values': [0.0005, 0.001]},  # Tighter threshold
        'force_reset': {'values': [True]},

        # ============== OPTIMIZER / SCHEDULER ==============
        # Lower LR strongly correlated with lower MSLE (+0.7 means high LR = bad)
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,   # Lower bound (was 5e-5)
            'max': 2e-4,   # Much lower upper bound (was 5e-4)
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 5e-4,  # Slightly reduced
        },
        # Lower factor = more aggressive decay, correlated with lower MSLE (-0.4)
        'lr_scheduler_factor': {
            'distribution': 'uniform',
            'min': 0.1,   # More aggressive (was 0.2)
            'max': 0.3,   # More aggressive (was 0.5)
        },
        'lr_scheduler_patience': {'values': [3, 4, 5]},  # Faster decay trigger
        'lr_scheduler_min_lr': {'values': [1e-7]},  # Lower floor
        'gradient_clip_val': {
            'distribution': 'uniform',
            'min': 0.5,
            'max': 1.0,
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
        'num_encoder_layers': {'values': [2, 3]},  # Slightly deeper
        'num_decoder_layers': {'values': [2, 3]},
        'decoder_output_dim': {'values': [32, 64]},
        'hidden_size': {'values': [128, 256]},  # Larger to capture signal
        
        'temporal_width_past': {'values': [4, 6, 8]},
        'temporal_width_future': {'values': [4, 6, 8]},
        'temporal_hidden_size_past': {'values': [32, 64]},
        'temporal_hidden_size_future': {'values': [32, 64]},
        'temporal_decoder_hidden': {'values': [64, 128]},
        
        # Regularization & normalization
        'use_layer_norm': {'values': [True]},  # Fixed True (correlation -0.2)
        'dropout': {'values': [0.35, 0.4, 0.45]},  # Higher dropout (correlation -0.3)
        'use_static_covariates': {'values': [True]},  # Fixed True for country-level
        'use_reversible_instance_norm': {'values': [False]},

        # ============== LOSS FUNCTION ==============
        # Optimized to maximize y_hat while minimizing MSLE
        # Key: High FN weight + Low FP weight → pushes predictions UP (avoids smoothing)
        'loss_function': {'values': ['WeightedPenaltyHuberLoss']},
        
        'zero_threshold': {'values': [0.01]},  # Fixed (data-dependent, not HP)
        
        # Lower delta = more L1-like, less smoothing (correlation -0.2)
        'delta': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.4,  # Lower range (was 0.1-1.0)
        },
        
        'non_zero_weight': {
            'distribution': 'uniform',
            'min': 5.0,   # Higher to focus on non-zero values
            'max': 10.0,
        },
        
        # LOW FP weight: Don't over-penalize false positives (correlation +0.3)
        # This allows model to predict higher without being punished
        'false_positive_weight': {
            'distribution': 'uniform',
            'min': 1.0,   # Lower (was 1.5-3.0)
            'max': 2.0,
        },
        
        # HIGH FN weight: Heavily penalize under-prediction (correlation -0.2)
        # This pushes predictions UP, avoiding smoothing/mean regression
        'false_negative_weight': {
            'distribution': 'uniform',
            'min': 5.0,   # Much higher (was 2.0-5.0)
            'max': 12.0,
        },
    }

    sweep_config['parameters'] = parameters
    return sweep_config