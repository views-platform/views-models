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
        'input_chunk_length': {'values': [36, 48, 60]},

        # Training basics
        'batch_size': {'values': [64, 96, 128]},
        'n_epochs': {'values': [300]},
        'early_stopping_patience': {'values': [10]},  # Allow learning of rare spikes

        # Optimizer / scheduler
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 4e-4,  # Cap prevents rapid collapse to zero baseline
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 5e-6,
            'max': 2e-4,
        },
        'lr_scheduler_factor': {
            'distribution': 'uniform',
            'min': 0.35,
            'max': 0.6,
        },
        'lr_scheduler_patience': {'values': [3, 5]},
        'lr_scheduler_min_lr': {'values': [1e-6]},

        # Scaling (fixed target scaler)
        'feature_scaler': {
            'values': ['RobustScaler']  # Consistent handling of outliers across logged conflict features
        },
        'target_scaler': {
            'values': ['RobustScaler']  # Single choice per requirement
        },
        'log_targets': {'values': [True]},
        'log_features': {
            'values': [["lr_ged_sb", "lr_ged_ns", "lr_ged_os", "lr_acled_sb", "lr_acled_os", "lr_ged_sb_tsum_24", 
                         "lr_splag_1_ged_sb", "lr_splag_1_ged_os", "lr_splag_1_ged_ns"]]
        },

        # Architecture (balanced capacity)
        'rnn_type': {'values': ['GRU', 'LSTM']},
        'hidden_dim': {'values': [64, 128, 192]},
        'n_rnn_layers': {'values': [1, 2]},
        'dropout': {'values': [0.15, 0.25, 0.35]},  # Lower dropout retains rare signal
        'activation': {'values': ['LeakyReLU', 'ReLU', 'Tanh']},
        'use_reversible_instance_norm': {'values': [True, False]},

        # Loss (spike-sensitive)
        'loss_function': {'values': ['WeightedPenaltyHuberLoss']},

        # Single zero threshold range (post log+RobustScaler); keeps 1–3 fatalities > 0 class
        'zero_threshold': {
            'distribution': 'log_uniform_values',
            'min': 3e-4,
            'max': 7e-3,
        },

        'false_positive_weight': {
            'distribution': 'uniform',
            'min': 1.3,
            'max': 3.2,  # Limit to avoid suppressing emergent positives
        },
        'false_negative_weight': {
            'distribution': 'uniform',
            'min': 2.5,
            'max': 5.2,  # Emphasize missing spikes
        },
        'non_zero_weight': {
            'distribution': 'uniform',
            'min': 4.0,
            'max': 8.0,
        },
        'delta': {
            'distribution': 'log_uniform_values',
            'min': 0.08,
            'max': 1.5,  # Mid-range keeps sensitivity to moderate spike errors
        },

        # Gradient stability
        'gradient_clip_val': {
            'distribution': 'uniform',
            'min': 0.6,
            'max': 1.4,
        },
    }

    sweep_config['parameters'] = parameters

    return sweep_config