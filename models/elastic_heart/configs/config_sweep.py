
def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    This configuration is "operational" so modifying it will change the search strategy, parameter ranges, and other settings for hyperparameter tuning aimed at optimizing model performance.

    Returns:
    - sweep_config (dict): A dictionary containing the configuration for hyperparameter sweeps, defining the methods and parameter ranges used to search for optimal hyperparameters.
    """

    sweep_config = {
        "method": "bayes",  
        "name": "elastic_heart_v3",
        "early_terminate": {"type": "hyperband", "min_iter": 15, "eta": 2},
    }

    # Example metric setup:
    metric = {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"}
    sweep_config["metric"] = metric

    # Example parameters setup:
    parameters_dict = {
        "steps": {"values": [[*range(1, 36 + 1, 1)]]},
        "input_chunk_length": {"values": [24, 36, 48]},

        "batch_size": {"values": [32, 64, 128]},  
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [20]},
        "early_stopping_min_delta": {"values": [0.01]},
        "gradient_clip_val": {"values": [0.1]}, 

        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-4,   
            "max": 5e-4,  
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-3,
        },
        
         
        'feature_scaler': {
            'values': ['MaxAbsScaler', 'MinMaxScaler']  
        },
        'target_scaler': {
            'values': ['MaxAbsScaler', 'MinMaxScaler', None] 
        },
        'log_targets': {
            'values': [True]  
        },
        
        # TSMixer-specific parameters
        "num_blocks": {"values": [2, 3, 4]},
        "ff_size": {"values": [64, 128, 256]},
        "hidden_size": {"values": [64, 128, 256]},
        "activation": {"values": ['ReLU', 'Tanh', 'GELU']},
        "dropout": {"values": [0.1, 0.2, 0.3]},  
        "norm_type": {"values": ["LayerNorm", "TimeBatchNorm2d"]},
        "normalize_before": {"values": [True]},  
        
        "loss_function": {"values": ["WeightedPenaltyHuberLoss"]},

        # TweedieLoss parameters
        # "p": {
        #     "distribution": "uniform",
        #     "min": 1.1,
        #     "max": 1.5,
        # },
      
        "non_zero_weight": {
            "distribution": "uniform",
            "min": 5.0,
            "max": 10.0,     
        },
        "false_positive_weight": {
            "distribution": "uniform",
            "min": 1.0,
            "max": 3.0,   
        },
        "false_negative_weight": {
            "distribution": "uniform",
            "min": 5.0,
            "max": 10.0,  
        },
        "delta": {
            "distribution": "uniform",
            "min": 0.05,   
            "max": 0.5,   
        },
        "zero_threshold": {
            "distribution": "uniform",
            "min": 0.01,   
            "max": 0.1,   
        },
        # "zero_weight": {
        #     "distribution": "uniform",
        #     "min": 0.2,
        #     "max": 0.8,
        # },
        # "count_weight": {
        #     "distribution": "uniform",
        #     "min": 0.2,
        #     "max": 0.8,
        # },
        # "nonzero_multiplier": {
        #     "values": [1.0]
        # },
    }

    sweep_config['parameters'] = parameters_dict

    return sweep_config
