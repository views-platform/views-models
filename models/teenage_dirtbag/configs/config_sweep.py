def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    This configuration is "operational" so modifying it will change the search strategy, parameter ranges, and other settings for hyperparameter tuning aimed at optimizing model performance.

    Returns:
    - sweep_config (dict): A dictionary containing the configuration for hyperparameter sweeps, defining the methods and parameter ranges used to search for optimal hyperparameters.
    """

    sweep_config = {
        'method': 'bayes',
        'name': 'teenage_dirtbag_tcn_focus',
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
        'input_chunk_length': {'values': [48, 60, 72]},  # Longer context for dilated convolutions

        # Training basics
        'batch_size': {'values': [64, 96, 128]},
        'n_epochs': {'values': [300]},
        'early_stopping_patience': {'values': [10]},

        # Optimizer / scheduler
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 4e-4,
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

        # Scaling and transformation
        'feature_scaler': {'values': ['RobustScaler']},
        'target_scaler': {'values': ['RobustScaler']},
        'log_targets': {'values': [True]},
        'log_features': {
            'values': [["lr_ged_sb", "lr_ged_ns", "lr_ged_os", "lr_acled_sb", "lr_acled_os", 
                       "lr_ged_sb_tsum_24", "lr_splag_1_ged_sb", "lr_splag_1_ged_os", "lr_splag_1_ged_ns"]]
        },

        # TCN specific architecture
        'kernel_size': {'values': [3, 5, 7]},  # Size of convolutional kernels
        'num_filters': {'values': [32, 64, 128]},  # Number of filters per convolutional layer
        'dilation_base': {'values': [2, 3, 4]},  # Base for exponential dilation
        'dropout': {'values': [0.1, 0.2, 0.3]},
        'use_reversible_instance_norm': {'values': [True, False]},  # Normalization technique

        # Loss function
        'loss_function': {'values': ['WeightedPenaltyHuberLoss']},

        # Loss function parameters - adjusted for log-transformed data
        'zero_threshold': {
            'distribution': 'uniform',
            'min': 0.1,  # Well above 0 to account for scaling
            'max': 0.5,  # Well below 0.693 to distinguish from smallest non-zero
        },
        'false_positive_weight': {
            'distribution': 'uniform',
            'min': 1.3,
            'max': 3.2,
        },
        'false_negative_weight': {
            'distribution': 'uniform',
            'min': 2.5,
            'max': 5.2,
        },
        'non_zero_weight': {
            'distribution': 'uniform',
            'min': 2.0,
            'max': 8.0,
        },
        'delta': {
            'distribution': 'log_uniform_values',
            'min': 0.08,
            'max': 1.5,
        },
    }

    sweep_config['parameters'] = parameters
    return sweep_config