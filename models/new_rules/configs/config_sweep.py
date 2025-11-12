def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    This configuration is "operational" so modifying it will change the search strategy, parameter ranges, and other settings for hyperparameter tuning aimed at optimizing model performance.

    Returns:
    - sweep_config (dict): A dictionary containing the configuration for hyperparameter sweeps, defining the methods and parameter ranges used to search for optimal hyperparameters.
    """

    sweep_config = {
        "method": "bayes",
        "name": "new_rules_nbeats",
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 5,
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
        
        # Input length: N-BEATS works well with longer lookback for trend/seasonality
        # Longer context helps detect conflict escalation patterns
        "input_chunk_length": {"values": [36, 48, 60, 72, 84]},
        
        # Training Configuration
        "batch_size": {"values": [64, 128, 256]},  # Higher batches for stability with sparse data
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [6]},
        
        # Learning rate: N-BEATS is less sensitive than transformers
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-5,  # Lower minimum for careful convergence
            "max": 1e-3,  # Conservative upper bound
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-3,  # Allow stronger regularization
        },
        
        # Scaling: Critical for zero-inflated fatality data
        'feature_scaler': {
            'values': ['MinMaxScaler']
        },
        'target_scaler': {
            'values': ['MinMaxScaler']  # LogTransform best for count data
        },
        
        
        # N-BEATS Architecture Parameters
        # Generic vs interpretable architecture
        "generic_architecture": {"values": [True, False]},  # False = interpretable (trend+seasonality)
        
        # More stacks = more hierarchical pattern learning
        "num_stacks": {"values": [3, 4, 5, 6, 7]},  # Expanded range for conflict patterns
        
        # Blocks per stack: Deeper = more capacity per stack
        "num_blocks": {"values": [1, 2, 3]},  # Keep shallow to moderate
        
        # Layer widths: Main capacity control
        # N-BEATS uses fully connected layers, so width matters more than depth
        "layer_widths": {"values": [128, 256, 512, 1024]},  # Increased max for complex patterns
        
        # Regularization
        "dropout": {"values": [0.1, 0.2, 0.3, 0.4, 0.5]},  # Wider range for sparse data
        "gradient_clip_val": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1.5  # Tighter range than original
        },
        
        # Activation functions: N-BEATS paper uses ReLU, but worth exploring
        "activation": {
            "values": [
                "ReLU",      # Original paper default
                "GELU",      # Smoother gradients
                "LeakyReLU", # Prevents dead neurons
                "ELU",       # Smooth for sparse data
            ]
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
        
        # False negatives: Missing actual conflicts
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
            "max": 5.0  # Wider range to handle outlier fatality spikes
        },
    }
    sweep_config["parameters"] = parameters_dict

    return sweep_config