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
        'name': 'cool_cat_tide_sweep_week_dylan',
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 15,
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
        'input_chunk_length': {'values': [36, 48, 60]},  # TiDE benefits from longer context
        'output_chunk_shift': {'values': [0]},

        # ============== TRAINING BASICS ==============
        'batch_size': {'values': [32, 64, 96]},
        'n_epochs': {'values': [300]},
        'early_stopping_patience': {'values': [10, 12, 15]},
        'early_stopping_min_delta': {'values': [0.001, 0.002]},
        'force_reset': {'values': [True]},

        # ============== OPTIMIZER / SCHEDULER ==============
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 5e-5,
            'max': 5e-4,
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-3,
        },
        'lr_scheduler_factor': {
            'distribution': 'uniform',
            'min': 0.2,
            'max': 0.5,
        },
        'lr_scheduler_patience': {'values': [4, 5, 6]},
        'lr_scheduler_min_lr': {'values': [1e-6]},
        'gradient_clip_val': {
            'distribution': 'uniform',
            'min': 0.5,
            'max': 1.0,
        },

        # ============== SCALING ==============
        # RobustScaler as default for features not in feature_scaler_map
        # AsinhTransform for targets (best for zero-inflated fatality counts)
        # feature_scaler_map applies AsinhTransform to zero-inflated features
        'feature_scaler': {'values': ['RobustScaler']},
        'target_scaler': {'values': ['AsinhTransform', 'RobustScaler']},
        'log_targets': {'values': [False]},
        'feature_scaler_map': {
            'values': [{
                "AsinhTransform": [
                    "ged_sb", "ged_ns", "ged_os", "acled_sb", "acled_os",
                    "ged_sb_tsum_24", "splag_1_ged_sb", "splag_1_ged_os", "splag_1_ged_ns",
                    "wdi_sm_pop_netm", "wdi_sm_pop_refg_or", "wdi_sp_dyn_imrt_fe_in", "wdi_ny_gdp_mktp_kd"
                ]
            }]
        },

        # ============== TiDE ARCHITECTURE ==============
        # Encoder-decoder structure with temporal projections
        'num_encoder_layers': {'values': [1, 2, 3]},
        'num_decoder_layers': {'values': [1, 2, 3]},
        'decoder_output_dim': {'values': [16, 32, 64]},
        'hidden_size': {'values': [64, 128, 256]},
        
        # Temporal width controls information compression
        'temporal_width_past': {'values': [4, 6, 8]},
        'temporal_width_future': {'values': [4, 6, 8]},
        'temporal_hidden_size_past': {'values': [16, 32, 64]},
        'temporal_hidden_size_future': {'values': [16, 32, 64]},
        'temporal_decoder_hidden': {'values': [32, 64, 128]},
        
        # Regularization & normalization
        'use_layer_norm': {'values': [True]},  # Critical for stability
        'dropout': {'values': [0.2, 0.3, 0.4]},
        'use_static_covariates': {'values': [True, False]},
        'use_reversible_instance_norm': {'values': [True]},  # Critical for non-stationary conflict

        # ============== LOSS FUNCTION ==============
        'loss_function': {'values': ['WeightedPenaltyHuberLoss']},
        
        # WeightedPenaltyHuberLoss parameters optimized for zero-inflated data
        'zero_threshold': {
            'distribution': 'uniform',
            'min': 0.01,
            'max': 0.1,
        },
        'delta': {
            'distribution': 'log_uniform_values',
            'min': 0.1,
            'max': 1.0,
        },
        'non_zero_weight': {
            'distribution': 'uniform',
            'min': 3.0,
            'max': 8.0,
        },
        'false_positive_weight': {
            'distribution': 'uniform',
            'min': 1.5,
            'max': 3.0,
        },
        'false_negative_weight': {
            'distribution': 'uniform',
            'min': 2.0,
            'max': 5.0,
        },
    }

    sweep_config['parameters'] = parameters
    return sweep_config