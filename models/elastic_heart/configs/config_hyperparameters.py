
def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.
    This configuration is "operational" so modifying these settings will impact the model's behavior during the training.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for training the model, which determine the model's behavior during the training phase.
    """
    
    hyperparameters = {
        'steps': [*range(1, 36 + 1, 1)],
        
        # classic-sweep-120
        "activation": "Tanh",
        "batch_size": 256,
        "delta": 3.789929511070176,
        "dropout": 0.4,
        "early_stopping_patience": 5,
        "false_negative_weight": 5.643294881537305,
        "false_positive_weight": 24.57766839638798,
        "feature_scaler": None,
        "ff_size": 256,
        "hidden_size": 128,
        "input_chunk_length": 36,
        "loss_function": "WeightedPenaltyHuberLoss",
        "lr": 0.0000548349633455129,
        "n_epochs": 10,
        "non_zero_weight": 17.16365112650754,
        "norm_type": "LayerNormNoBias",
        "normalize_before": False,
        "num_blocks": 1,
        "target_scaler": "YeoJohnsonTransform",
        "weight_decay": 0.005893888408985461,
        "zero_threshold": 0.2163136317477652,

        "num_samples": 1,
        "mc_dropout": True,
    }
    return hyperparameters
