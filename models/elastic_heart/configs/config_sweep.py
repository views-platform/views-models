def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    This configuration is "operational" so modifying it will change the search strategy, parameter ranges, and other settings for hyperparameter tuning aimed at optimizing model performance.

    Returns:
    - sweep_config (dict): A dictionary containing the configuration for hyperparameter sweeps, defining the methods and parameter ranges used to search for optimal hyperparameters.
    """

    sweep_config = {
        "method": "bayes",
        "name": "elastic_heart_tsmixer",
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 5,  # TSMixer converges reasonably fast
            "eta": 2
        },
    }

    metric = {
        "name": "time_series_wise_msle_mean_sb",
        "goal": "minimize"
    }
    sweep_config["metric"] = metric

    parameters_dict = {
        # Temporal Configuration
        "steps": {"values": [[*range(1, 36 + 1, 1)]]},
        
        # Input length: TSMixer benefits from moderate to long context
        "input_chunk_length": {"values": [36, 48, 60, 72, 84]},
        
        # Training Configuration
        "batch_size": {"values": [64, 128, 256]},  # Reasonable range for memory/stability
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [6]},
        
        # Learning rate: TSMixer is stable like other MLP-based models
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-3,
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-3,
        },
        
        # Scaling: Critical for zero-inflated fatality data
        'feature_scaler': {
            'values': ['MinMaxScaler']
        },
        'target_scaler': {
            'values': ['MinMaxScaler']  # LogTransform best for count data
        },
        
        # TSMixer Architecture Parameters - CORE HYPERPARAMETERS
        
        # Number of mixer blocks: Controls model depth
        # More blocks = more mixing of time/feature dimensions
        "num_blocks": {"values": [2, 3, 4, 5, 6]},  # Removed 1 (too shallow)
        
        # Feed-forward size: Hidden dimension in MLP layers
        # Controls capacity of time/feature mixing
        "ff_size": {"values": [32, 64, 128, 256, 512]},  # Removed extremes
        
        # Hidden size: Projection dimension after mixing
        # Should be comparable to or smaller than ff_size
        "hidden_size": {"values": [32, 64, 128, 256, 512]},  # Removed extremes
        
        # Activation functions: TSMixer paper uses ReLU, but worth exploring
        "activation": {
            "values": [
                "ReLU",      # Original paper default
                "GELU",      # Smoother gradients, modern standard
                "LeakyReLU", # Prevents dead neurons
                "ELU",       # Smooth, good for sparse data
            ]
        },  # Removed sigmoid/tanh (saturate), SELU/PReLU (less common)
        
        # Dropout: TSMixer can overfit on sparse data
        "dropout": {"values": [0.1, 0.2, 0.3, 0.4, 0.5]},  # Wide range
        
        # Normalization type: Critical for mixing stability
        "norm_type": {
            "values": [
                "LayerNorm",           # Standard, proven
                "LayerNormNoBias",     # Simpler variant
                "TimeBatchNorm2d",     # Temporal-aware normalization
            ]
        },  # Removed RMSNorm (not standard in TSMixer)
        
        # Normalize before vs after mixing: Architectural choice
        "normalize_before": {"values": [True, False]},
        
        # Loss Function Configuration - Critical for zero-inflated data
        "loss_function": {"values": ["WeightedPenaltyHuberLoss"]},
        
        # Zero threshold: What counts as "zero" in fatality data
        'zero_threshold': {
            'distribution': 'log_uniform_values',
            'min': 0.0001,  # ~1-10 fatalities in typical scaled space
            'max': 0.01,    # ~100-250 fatalities in typical scaled space
        },
        
        # False positives: Predicting conflict when there is none
        "false_positive_weight": {
            "distribution": "uniform",
            "min": 2.0,
            "max": 20.0,
        },
        
        # False negatives: Missing actual conflicts (CRITICAL)
        "false_negative_weight": {
            "distribution": "uniform",
            "min": 5.0,
            "max": 30.0,
        },
        
        # Non-zero weight: General importance of conflict events
        "non_zero_weight": {
            "distribution": "uniform",
            "min": 3.0,
            "max": 20.0,
        },
        
        # Huber delta: Transition point between L2 and L1 loss
        "delta": {
            "distribution": "uniform",
            "min": 0.01,
            "max": 5.0,
        },
    }

    sweep_config['parameters'] = parameters_dict

    return sweep_config