def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    This configuration is "operational" so modifying it will change the search strategy, parameter ranges, and other settings for hyperparameter tuning aimed at optimizing model performance.

    Returns:
    - sweep_config (dict): A dictionary containing the configuration for hyperparameter sweeps, defining the methods and parameter ranges used to search for optimal hyperparameters.
    """

    sweep_config = {
        'method': 'bayes',
        'name': 'heat_waves_tft_focus',
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
        # Temporal horizon & context
        'steps': {'values': [[*range(1, 36 + 1)]]},
        'input_chunk_length': {'values': [36, 48, 60]},
        'output_chunk_shift': {'values': [0, 1, 2]},

        # Training basics - FIXED: smaller batches, more epochs
        'batch_size': {'values': [32, 64, 96, 128]},  # Reduced from 256
        'n_epochs': {'values': [300]},
        'early_stopping_patience': {'values': [15, 20, 25]},
        'early_stopping_min_delta': {'values': [0.001, 0.005, 0.01]},

        # Optimizer / scheduler - FIXED: lower learning rates
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,  # Much lower to prevent explosion
            'max': 1e-4,  # Lower max for stability
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-3,
        },
        'lr_scheduler_factor': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.5,
        },
        'lr_scheduler_patience': {'values': [3, 5, 7]},
        'lr_scheduler_min_lr': {'values': [1e-6, 1e-5]},

        # Scaling and transformation
        'feature_scaler': {'values': ['RobustScaler']},
        'target_scaler': {'values': ['RobustScaler']},
        'log_targets': {'values': [True]},
        'log_features': {
            'values': [
                ["lr_ged_sb", "lr_ged_ns", "lr_ged_os", "lr_acled_sb", "lr_acled_os", 
                 "lr_ged_sb_tsum_24", "lr_splag_1_ged_sb", "lr_splag_1_ged_os", "lr_splag_1_ged_ns", 
                 "lr_wdi_sm_pop_netm", "lr_wdi_sm_pop_refg_or", "lr_wdi_sp_dyn_imrt_fe_in", "lr_wdi_ny_gdp_mktp_kd",
                 ]
            ]
        },

        # TFT specific architecture - FIXED: more conservative
        'hidden_size': {'values': [32, 64, 128]},  # Reduced from 256
        'lstm_layers': {'values': [1, 2]},  # Reduced from 4
        'num_attention_heads': {'values': [2, 4]},  # Reduced from 8
        'dropout': {'values': [0.1, 0.2, 0.3]},  # Reduced from 0.3
        'full_attention': {'values': [True, False]},
        'feed_forward': {'values': ['GatedResidualNetwork', 'Linear']},
        'add_relative_index': {'values': [True, False]},
        'use_static_covariates': {'values': [True]},
        'norm_type': {'values': ['LayerNorm', 'RMSNorm']},
        'force_reset': {'values': [True]},

        # Loss function
        'loss_function': {'values': ['WeightedPenaltyHuberLoss']},

        # Loss function parameters
        'zero_threshold': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.5,
        },
        'false_positive_weight': {
            'distribution': 'uniform',
            'min': 1.0,
            'max': 3.0,
        },
        'false_negative_weight': {
            'distribution': 'uniform',
            'min': 3.0,
            'max': 8.0,
        },
        'non_zero_weight': {
            'distribution': 'uniform',
            'min': 3.0,
            'max': 7.0,
        },
        'delta': {
            'distribution': 'log_uniform_values',
            'min': 0.1,
            'max': 1.0,
        },
        
        # Gradient clipping - CRITICAL: added to prevent explosion
        'gradient_clip_val': {
            'distribution': 'uniform',
            'min': 0.5,
            'max': 1.0,
        },
    }

    sweep_config['parameters'] = parameters
    return sweep_config