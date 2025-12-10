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
        "attention_dropout": 0.4,
        "batch_size": 128,
        "delta": 2.174105924517258,
        "early_stopping_patience": 5,
        "false_negative_weight": 9.143637302405882,
        "false_positive_weight": 8.173608105213063,
        "feature_scaler": "MaxAbsScaler",
        "feed_forward": "GELU",
        "full_attention": False,
        "hidden_size": 128,
        "input_chunk_length": 60,
        "loss_function": "WeightedPenaltyHuberLoss",
        "lr": 0.0001,
        "lstm_layers": 1,
        "n_epochs": 2,
        "non_zero_weight": 11.810582310673835,
        "num_attention_heads": 4,
        "target_scaler": "RobustScaler",
        "use_reversible_instance_norm": False,
        "weight_decay": 0.00002389357463410524,
        "zero_threshold": 0.0881212976698389,
        "num_samples": 1,
        "mc_dropout": False,
    }
    return hyperparameters
