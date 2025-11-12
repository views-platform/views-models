def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    This configuration is "operational" so modifying it will change the search strategy, parameter ranges, and other settings for hyperparameter tuning aimed at optimizing model performance.

    Returns:
    - sweep_config (dict): A dictionary containing the configuration for hyperparameter sweeps, defining the methods and parameter ranges used to search for optimal hyperparameters.
    """

    sweep_config = {
        "method": "bayes",
        "name": "heat_waves_tft",
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
        
        # Input length: TFT benefits from longer context for attention
        "input_chunk_length": {"values": [36, 48, 60, 72]},  # Removed 24 (too short)
        
        # Training Configuration
        "batch_size": {"values": [64, 128, 256]},  # Catalog default 256, removed 512
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [5]},
        
        # Learning rate: TFT is sensitive, needs careful tuning
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-5,  # More conservative minimum
            "max": 1e-3,  # Catalog default 3e-4, explore around it
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-3,  # Catalog default 1e-3
        },
        
        # Scaling: Critical for zero-inflated fatality data
        'feature_scaler': {
            'values': ['MaxAbsScaler', 'MinMaxScaler']
        },
        'target_scaler': {
            'values': ['MinMaxScaler', 'MaxAbsScaler']  # LogTransform best for count data
        },
        
        # TFT Architecture Parameters - CORE HYPERPARAMETERS
        
        # Hidden size: Main capacity control (LSTM hidden state dimension)
        # TFT paper uses 16-256 range
        "hidden_size": {"values": [32, 64, 128, 256, 512]},
        
        # LSTM layers: Temporal feature extractor depth
        # More layers = more capacity but slower training
        "lstm_layers": {"values": [1, 2, 3]},
        
        # Attention heads: Multi-head attention mechanism
        # Must divide hidden_size evenly (e.g., 256/4=64 per head)
        "num_attention_heads": {"values": [1, 2, 4, 8]},  # Power of 2 for efficiency
        
        # Full attention: Whether to use full or sparse attention
        # Full = quadratic memory, sparse = linear memory
        "full_attention": {"values": [False]},  # Set to False only - full is too expensive
        
        # Feed-forward network type: Critical for TFT's gating mechanisms
        "feed_forward": {
            "values": [
                "GatedResidualNetwork",  # TFT paper default - BEST for sparse data
                "GLU",                    # Gated Linear Unit - good alternative
                "GEGLU",                  # Gated GELU - modern variant
                "ReLU",                   # Simple baseline
            ]
        },
        
        # Attention dropout: Prevents attention overfitting
        "attention_dropout": {"values": [0.0, 0.1, 0.2, 0.3]},
        
        # Reversible Instance Normalization
        "use_reversible_instance_norm": {"values": [True, False]},
        
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
            "max": 20.0,  # Increased from 15 to match other models
        },
        
        # False negatives: Missing actual conflicts
        "false_negative_weight": {
            "distribution": "uniform",
            "min": 5.0,
            "max": 30.0,
        },
        
        # Non-zero weight: General importance of conflict events
        "non_zero_weight": {
            "distribution": "uniform",
            "min": 3.0,  # Increased from 1.0
            "max": 20.0,  # Increased from 15
        },
        
        # Huber delta: Transition point between L2 and L1 loss
        "delta": {
            "distribution": "uniform",
            "min": 0.01,  # Increased from 0.01
            "max": 5.0  # Same as other models
        },
    }
    sweep_config["parameters"] = parameters_dict

    return sweep_config