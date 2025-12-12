def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    This configuration is "operational" so modifying it will change the search strategy, parameter ranges, and other settings for hyperparameter tuning aimed at optimizing model performance.

    Returns:
    - sweep_config (dict): A dictionary containing the configuration for hyperparameter sweeps, defining the methods and parameter ranges used to search for optimal hyperparameters.
    """

    sweep_config = {
        "method": "bayes",
        "name": "heat_waves_tft_new_hope",
        "early_terminate": {"type": "hyperband", "min_iter": 10, "eta": 2},
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # Temporal horizon & context
        "steps": {"values": [[*range(1, 36 + 1)]]},
        "input_chunk_length": {"values": [18, 24, 48]},
        "output_chunk_shift": {"values": [0]},
        # Training basics - FIXED: smaller batches, more epochs
        "batch_size": {"values": [256, 512]},  # Reduced from 256
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [15]},
        "early_stopping_min_delta": {"values": [0.001]},
        # Optimizer / scheduler - FIXED: lower learning rates
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-5,  # Much lower to prevent explosion
            "max": 5e-3,  # Lower max for stability
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-3,
        },
        "lr_scheduler_factor": {
            "distribution": "uniform",
            "min": 0.2,
            "max": 0.5,
        },
        "lr_scheduler_patience": {"values": [3, 5]},
        "lr_scheduler_min_lr": {"values": [1e-6]},
        # Scaling and transformation
        "feature_scaler": {"values": ["RobustScaler", "MinMaxScaler"]},
        "target_scaler": {"values": ["RobustScaler"]},
        "log_targets": {"values": [True]},
        "log_targets": {"values": [True]},
        # TFT specific architecture - FIXED: more conservative
        "hidden_size": {"values": [2, 4, 8]},  # Reduced from 256
        "lstm_layers": {"values": [1, 2]},  # Reduced from 4
        "num_attention_heads": {"values": [2, 4]},  # Reduced from 8
        "dropout": {"values": [0.1, 0.2, 0.3]},  # Reduced from 0.3
        "full_attention": {"values": [True]},
        "feed_forward": {
            "values": ["ReGLU", "GEGLU", "GLU"]
        },  # dylan: 'GLU', 'Bilinear', 'ReGLU', 'GEGLU', 'SwiGLU', 'ReLU', 'GELU', 'GatedResidualNetwork'
        "add_relative_index": {"values": [True]},
        "use_static_covariates": {"values": [True]},
        "norm_type": {"values": ["LayerNorm"]},  # dylan: 'LayerNorm', 'RMSNorm'
        "force_reset": {"values": [True]},
        # Loss function
        "loss_function": {"values": ["WeightedPenaltyHuberLoss"]},
        # Loss function parameters
        "zero_threshold": {
            "distribution": "uniform",
            "min": -0.3,
            "max": -0.1,
        },
        "false_positive_weight": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 2.0,
        },
        "false_negative_weight": {
            "distribution": "uniform",
            "min": 4.0,
            "max": 10.0,
        },
        "non_zero_weight": {
            "distribution": "uniform",
            "min": 3.0,
            "max": 8.0,
        },
        "delta": {
            "distribution": "log_uniform_values",
            "min": 0.2,
            "max": 1.0,
        },
        # Gradient clipping - CRITICAL: added to prevent explosion
        "gradient_clip_val": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1.0,
        },
    }

    sweep_config["parameters"] = parameters
    return sweep_config
