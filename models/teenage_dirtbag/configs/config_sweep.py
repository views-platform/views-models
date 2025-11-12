def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    This configuration is "operational" so modifying it will change the search strategy, parameter ranges, and other settings for hyperparameter tuning aimed at optimizing model performance.

    Returns:
    - sweep_config (dict): A dictionary containing the configuration for hyperparameter sweeps, defining the methods and parameter ranges used to search for optimal hyperparameters.
    """

    sweep_config = {
        "method": "bayes",
        "name": "teenage_dirtbag_tcn",
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 6,  # TCN converges faster than transformers
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
        
        # Input length: TCN receptive field critical
        # Receptive field = 1 + 2 * (kernel_size - 1) * sum(dilation_base^i for i in range(num_filters))
        # For k=3, d=2, layers=6: RF = 1 + 2*(3-1)*(2^0+2^1+2^2+2^3+2^4+2^5) = 253
        # k=3, d=2, layers=6: RF = 253 months (covers ~21 years)
        # k=5, d=3, layers=5: RF = 1211 months (covers ~100 years)
        # k=2, d=2, layers=7: RF = 255 months
        "input_chunk_length": {"values": [36, 48, 60, 72, 84]},
        
        # Training Configuration
        "batch_size": {"values": [64, 128, 256]},  # TCN handles larger batches well
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [6]},
        
        # Learning rate: TCN is stable, can use moderate LR
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-3,  # Conservative upper bound
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-3,  # Allow strong regularization
        },
        
        # Scaling: Critical for zero-inflated fatality data
        'feature_scaler': {
            'values': ['MaxAbsScaler', 'MinMaxScaler']
        },
        'target_scaler': {
            'values': ['LogTransform', 'MinMaxScaler', 'MaxAbsScaler']  # LogTransform best for count data
        },
        
        
        # TCN Architecture Parameters - CORE HYPERPARAMETERS
        
        # Number of convolutional layers: Controls receptive field
        # More layers = longer memory, but risk of gradient issues
        "num_filters": {"values": [3, 4, 5, 6, 7]},  # 3-7 is sweet spot for TCN
        
        # Kernel size: Local pattern detection
        # Larger = smoother patterns, smaller = more detail
        "kernel_size": {"values": [2, 3, 4, 5, 7]},  # Added 7, removed 6 (even kernels less common)
        
        # Dilation base: Controls exponential receptive field growth
        # 2 = standard (1,2,4,8,16...), 3 = aggressive (1,3,9,27...)
        "dilation_base": {"values": [2, 3, 4]},  # 2 is standard, 3-4 for longer dependencies
        
        # Weight normalization: Stabilizes training but can slow convergence
        # NOTE: Catalog has weight_norm commented out as BUG - keeping in sweep anyway
        "weight_norm": {"values": [False]},  # Set to False only due to noted bug
        
        # Reversible Instance Norm: Good for non-stationary data
        "use_reversible_instance_norm": {"values": [True, False]},
        
        # Regularization
        "dropout": {"values": [0.1, 0.2, 0.25, 0.3, 0.4]},  # Catalog default 0.25, explore around it
        
        "gradient_clip_val": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1.5  # TCN can have vanishing gradients with deep networks
        },
        
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
            "max": 20.0,  # High penalty to avoid false alarms
        },
        
        # False negatives: Missing actual conflicts (CRITICAL)
        "false_negative_weight": {
            "distribution": "uniform",
            "min": 5.0,
            "max": 30.0,  # Very high penalty - missing conflicts is critical
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
            "max": 5.0  # Allow larger deltas for fatality spikes
        },
    }
    sweep_config["parameters"] = parameters_dict

    return sweep_config