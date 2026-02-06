
def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.
    This configuration is "operational" so modifying these settings will impact the model's behavior during the training.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for training the model, which determine the model's behavior during the training phase.
    """
    
    hyperparameters = {
        'steps': [*range(1, 36 + 1, 1)],
        
        "log_targets": True,  
        "feature_scaler": "MinMaxScaler", 
        "target_scaler": None,  

        # Model architecture 
        "activation": "ReLU",
        "batch_size": 128,
        "dropout": 0.3,
        "ff_size": 128,
        "hidden_size": 128,
        "input_chunk_length": 36, 
        "norm_type": "TimeBatchNorm2d",
        "normalize_before": True,
        "num_blocks": 2,
        
        # Training params 
        "n_epochs": 300,
        "lr": 0.0004409588821525886,
        "weight_decay": 0.00000651940711962416,
        "early_stopping_patience": 20,
        "early_stopping_min_delta": 0.01,
        "gradient_clip_val": 0.1, 
        
        # Loss function: WeightedPenaltyHuberLoss
        "loss_function": "WeightedPenaltyHuberLoss",
        "non_zero_weight": 9.928451099354408,
        "false_positive_weight": 2.1913572944255617,
        "false_negative_weight": 7.396545661266665,
        "delta": 0.15094303659874916,
        "zero_threshold": 0.09818129846394924,

        # Prediction params
        "num_samples": 1,
        # "mc_dropout": True
    }
    return hyperparameters

   
