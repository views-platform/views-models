def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.
    This configuration is "operational" so modifying these settings will impact the model's behavior during the training.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for training the model, which determine the model's behavior during the training phase.
    """

    hyperparameters = {
        "steps": [*range(1, 36 + 1, 1)],
        # royal-sweep-63
        # "batch_size": 256,
        # "attention_dropout": 0.4,
        # "delta": 0.6964465184869596,
        # "early_stopping_patience": 5,
        # "false_negative_weight": 11.588512629493511,
        # "false_positive_weight": 2.2360438632711896,
        # "feature_scaler": "MaxAbsScaler",
        # "feed_forward": "Bilinear",
        # "full_attention": False,
        # "hidden_size": 512,
        # "input_chunk_length": 48,
        # "loss_function": "WeightedPenaltyHuberLoss",
        # "lr": 0.00001618100811761906,
        # "lstm_layers": 2,
        # "n_epochs": 300,
        # "non_zero_weight": 2.35346610056854,
        # "num_attention_heads": 4,
        # "target_scaler": None,
        # "use_reversible_instance_norm": True,
        # "weight_decay": 0.00001339740219226816,
        # "zero_threshold": 0.013101193138707702,
        # rural-sweep-287
        "attention_dropout": 0.3,
        "batch_size": 256,
        "delta": 0.9319640217513432,
        "early_stopping_patience": 3,
        "false_negative_weight": 0.9319640217513432,
        "false_positive_weight": 0.9319640217513432,
        "feature_scaler": "MinMaxScaler",
        "feed_forward": "GLU",  # GEGLU
        "full_attention": True,
        "log_targets": True,
        "hidden_size": 32,
        "input_chunk_length": 18,
        "loss_function": "WeightedPenaltyHuberLoss",
        "lr": 0.0018480545700327537,
        "lstm_layers": 4,
        "n_epochs": 300,
        "non_zero_weight": 5.998831066834101,
        "num_attention_heads": 4,
        "target_scaler": "RobustScaler",
        "use_reversible_instance_norm": True,  # True
        "weight_decay": 4.922709397770848e-05,
        "zero_threshold": -0.2263667462473728,
        "num_samples": 1,
        "use_static_covariates": True,
        "mc_dropout": True,
        "random_state": 42,
    }
    return hyperparameters
