import math

def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    This configuration is "operational" so modifying it will change the search strategy, parameter ranges, and other settings for hyperparameter tuning aimed at optimizing model performance.

    Returns:
    - sweep_config (dict): A dictionary containing the configuration for hyperparameter sweeps, defining the methods and parameter ranges used to search for optimal hyperparameters.
    """

    sweep_config = {
        'method': 'bayes',
        'name': 'dancing_queen_rnn_focus',
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
        'input_chunk_length': {'values': [36, 48, 60, 72]},
        'output_chunk_shift': {'values': [0, 1, 2]},

        # Training basics - more epochs to learn rare patterns
        'batch_size': {'values': [32, 64, 96, 128]},
        'n_epochs': {'values': [300]},
        'early_stopping_patience': {'values': [15, 20, 25]},  # More patience for rare events
        'early_stopping_min_delta': {'values': [0.001, 0.005, 0.01]},

        # Optimizer / scheduler
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
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

        # RNN specific architecture - FIXED: increased capacity
        'rnn_type': {'values': ['LSTM', 'GRU']},  # Both tested
        'hidden_dim': {'values': [128, 256, 512]},  # Increased from 256
        'n_rnn_layers': {'values': [2, 3, 4]},  # More layers for complexity
        'dropout': {'values': [0.1, 0.2, 0.3]},  # Moderate dropout
        'activation': {'values': ['ReLU', 'LeakyReLU', 'GELU']},  # GELU for rare events
        'use_reversible_instance_norm': {'values': [False]},  # disabled to prevent smoothing (this is a big damn problem ffs)
        'random_state': {'values': [42, 123, 2023]},
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
        
        # Gradient clipping
        'gradient_clip_val': {
            'distribution': 'uniform',
            'min': 0.5,
            'max': 1.0,
        },
    }

    sweep_config['parameters'] = parameters
    return sweep_config