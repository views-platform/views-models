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
        "attention_dropout": 0.1,
        "batch_size": 512,
        "delta": 1,
        "early_stopping_patience": 5,
        "false_negative_weight": 3,
        "false_positive_weight": 1.5,
        "feature_scaler": "MinMaxScaler",
        "feed_forward": "SwiGLU",  # GEGLU
        "full_attention": True,
        "log_targets": True,
        "hidden_size": 4,
        "input_chunk_length": 18,
        "loss_function": "WeightedPenaltyHuberLoss",
        "lr": 3e-5,
        "lstm_layers": 4,
        "n_epochs": 100,
        "non_zero_weight": 7,
        "num_attention_heads": 2,
        "target_scaler": "RobustScaler",
        "use_reversible_instance_norm": True,  # True
        "weight_decay": 1e-5,
        "zero_threshold": -0.1,
        "num_samples": 1,
        "mc_dropout": True,
    }
    return hyperparameters
