def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    This configuration is "operational" so modifying it will change the search strategy, parameter ranges, and other settings for hyperparameter tuning aimed at optimizing model performance.

    Returns:
    - sweep_config (dict): A dictionary containing the configuration for hyperparameter sweeps, defining the methods and parameter ranges used to search for optimal hyperparameters.
    """

    sweep_config = {
        'method': 'bayes',
        'name': 'cool_cat_tide',
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 8,  # Allow TiDE's temporal features to stabilize
            'eta': 2
        },
    }

    metric = {
        'name': 'time_series_wise_msle_mean_sb',
        'goal': 'minimize'
    }
    sweep_config['metric'] = metric

    parameters_dict = {
        # Temporal Configuration
        'steps': {'values': [[*range(1, 36 + 1, 1)]]},
        
        # Input length: TiDE works well with moderate history for sparse data
        # Too long = overfitting to zeros, too short = missing conflict patterns
        'input_chunk_length': {'values': [36, 48, 60, 72]},
        
        # Training Configuration
        'batch_size': {'values': [64, 128]},  # Larger batches help with sparse data stability (I hope lol)
        'n_epochs': {'values': [300]},
        'early_stopping_patience': {'values': [8]},  # Increased for sparse data convergence
        
        # Learning rate: TiDE prefers slightly higher LR than transformers
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 1e-4,
            'max': 5e-3,  # Upper bound increased based on catalog default
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-4  # Lower range to avoid over-regularizing sparse signals
        },
        
        # Scaling: Critical for zero-inflated data
        'feature_scaler': {
            'values': ['MinMaxScaler']
        },
        'target_scaler': {
            'values': ['MinMaxScaler']  # LogTransform best for count data
        },
        
        
        # TiDE Architecture Parameters
        # Encoder/Decoder layers: TiDE is designed to be shallow
        'num_encoder_layers': {'values': [1, 2]},  # Original paper uses 1-2
        'num_decoder_layers': {'values': [1, 2]},
        
        # Hidden size: Main capacity control for TiDE
        'hidden_size': {'values': [64, 128, 256, 512]},
        
        # Decoder output: Should be smaller than hidden_size
        'decoder_output_dim': {'values': [8, 16, 32, 64]},
        
        # These capture time-varying patterns in conflict data
        'temporal_width_past': {'values': [2, 4, 8]},  # How many past features to extract
        'temporal_width_future': {'values': [2, 4, 8]},  # How many future features to project
        'temporal_decoder_hidden': {'values': [16, 32, 64, 128]},
        
        # Regularization
        'dropout': {'values': [0.2, 0.3, 0.4, 0.5]},  # Higher range for sparse data
        'use_layer_norm': {'values': [True, False]},
        'gradient_clip_val': {
            'distribution': 'uniform',
            'min': 0.5,
            'max': 2.0  # Increased for sparse data gradient instability
        },
        
        # Loss Function Configuration - Critical for zero-inflated data
        "loss_function": {'values': ["WeightedPenaltyHuberLoss"]},
        
        # Zero threshold: What counts as "zero" in fatality data
        'zero_threshold': {
            'distribution': 'log_uniform_values',
            'min': 0.0001,  # ~1-10 fatalities in typical scaled space
            'max': 0.01,    # ~100-250 fatalities in typical scaled space
        },

        # False positives: Predicting conflict when there is none
        # Lower weight = more conservative predictions
        "false_positive_weight": {
            "distribution": "uniform",
            "min": 1.5,  # At least 1.5x base weight
            "max": 5.0,  # Up to 5x base weight
        },
        
        # False negatives: Missing actual conflicts (CRITICAL)
        "false_negative_weight": {
            "distribution": "uniform",
            'min': 2.0,  # At least 2x base weight (FN worse than FP)
            'max': 8.0,  # Up to 8x base weight
        },
        
        # Non-zero weight: General importance of conflict events
        'non_zero_weight': {
            'distribution': 'uniform',
            'min': 3.0,
            'max': 15.0,
        },
        
        # Huber delta: Transition point between L2 and L1 loss
        'delta': {
            'distribution': 'log_uniform_values',
            'min': 0.01,
            'max': 3.0  # Allow larger deltas for fatality counts
        },
    }
    sweep_config['parameters'] = parameters_dict

    return sweep_config