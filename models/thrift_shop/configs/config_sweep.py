def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    This configuration is "operational" so modifying it will change the search strategy, parameter ranges, and other settings for hyperparameter tuning aimed at optimizing model performance.

    Returns:
    - sweep_config (dict): A dictionary containing the configuration for hyperparameter sweeps, defining the methods and parameter ranges used to search for optimal hyperparameters.
    """

    sweep_config = {
        "method": "bayes",  # Other options: 'grid', 'bayes'
        "name": "thrift_shop",
        "early_terminate": {"type": "hyperband", "min_iter": 5, "eta": 2},
    }

    # Example metric setup:
    metric = {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"}
    sweep_config["metric"] = metric

    # Example parameters setup:
    parameters_dict = {
        "steps": {"values": [[*range(1, 36 + 1, 1)]]},
        "input_chunk_length": {"values": [24, 36, 48, 60, 72]},
        "batch_size": {"values": [32, 64, 128, 256, 512]},
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-3,
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-2,
        },
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [5]},

        'hidden_size': {'values': [1, 2, 4, 6, 8, 16, 32, 64, 128, 256, 512]},
        'lstm_layers': {'values': [1, 2, 3, 4]},
        'num_attention_heads': {'values': [1, 2, 4, 8]},
        'full_attention': {'values': [True, False]},
        'feed_forward': {
            'values': [
                'GatedResidualNetwork', 
                'GLU', 
                'Bilinear', 
                'ReGLU', 
                'GEGLU', 
                'SwiGLU', 
                'ReLU', 
                'GELU'
            ]
        },
        'feature_scaler': {
            'values': ['MinMaxScaler', 'MaxAbsScaler',  None]
        },
        'target_scaler': {
            'values': ['MinMaxScaler', 'MaxAbsScaler',  None]
        },
        'attention_dropout': {'values': [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]},
        'use_reversible_instance_norm': {'values': [True, False]},

        "loss_function": {"values": ["WeightedPenaltyHuberLoss"]},
        
        # Common loss parameters
        "zero_threshold": {"distribution": "uniform", "min": 0.001, "max": 0.3},
        "false_positive_weight": {
            "distribution": "uniform",
            "min": 0.1,
            "max": 15.0,
        },
        "false_negative_weight": {
            "distribution": "uniform",
            "min": 0.1,
            "max": 15.0,
        },
        "non_zero_weight": {
            "distribution": "uniform",
            "min": 1.0,
            "max": 15.0,
        },
        "delta": {"distribution": "uniform", "min": 0.01, "max": 5.0},
    }
    sweep_config["parameters"] = parameters_dict

    return sweep_config
