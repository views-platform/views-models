
def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    This configuration is "operational" so modifying it will change the search strategy, parameter ranges, and other settings for hyperparameter tuning aimed at optimizing model performance.

    Returns:
    - sweep_config (dict): A dictionary containing the configuration for hyperparameter sweeps, defining the methods and parameter ranges used to search for optimal hyperparameters.
    """

    sweep_config = {
        "method": "bayes",  # Other options: 'grid', 'bayes'
        "name": "elastic_heart",
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
        "early_stopping_patience": {"values": [6]},
        'feature_scaler': {
            'values': ['MinMaxScaler', 'MaxAbsScaler',  None]
        },
        'target_scaler': {
            'values': ['MinMaxScaler', 'MaxAbsScaler',  None]
        },
        
        # TSMixer-specific parameters
        "num_blocks": {"values": [1, 2, 3, 4, 5, 6]},
        "ff_size": {"values": [2, 4, 8, 16, 32, 64, 128, 256]},
        "hidden_size": {"values": [2, 4, 8, 16, 32, 64, 128, 256]},
        "activation": {"values": ['ReLU', 'RReLU', 'PReLU', 'ELU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'Sigmoid', 'GELU']},
        "dropout": {"values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]},
        "norm_type": {"values": ["LayerNorm", "LayerNormNoBias", "TimeBatchNorm2d"]},
        "normalize_before": {"values": [True, False]},
        
        # Loss function configuration
        "loss_function": {"values": ["WeightedPenaltyHuberLoss"]},
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

    sweep_config['parameters'] = parameters_dict

    return sweep_config
