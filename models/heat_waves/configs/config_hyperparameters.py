def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.
    This configuration is "operational" so modifying these settings will impact the model's behavior during the training.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for training the model, which determine the model's behavior during the training phase.
    """
    
    hyperparameters = {
        'steps': [*range(1, 36 + 1, 1)],
        "time_steps": 36,


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
        "hidden_size": 512,
        "input_chunk_length": 60,
        "loss_function": "WeightedPenaltyHuberLoss",
        "lr": 0.000021841168537191,
        "lstm_layers": 3,
        "n_epochs": 300,
        "non_zero_weight": 11.810582310673835,
        "num_attention_heads": 2,
        "target_scaler": "MinMaxScaler",
        "use_reversible_instance_norm": False,
        "weight_decay": 0.00002389357463410524,
        "zero_threshold": 0.0881212976698389,

        "output_chunk_length": 36,
        "output_chunk_shift": 0,
        "dropout": 0.3,
        "add_relative_index": True,
        "use_static_covariates": True,
        "norm_type": "LayerNorm",
        "skip_interpolation": False,
        "hidden_continuous_size": 8,

        "random_state": 1,
        "optimizer_cls": "Adam",
        "lr_scheduler_factor": 0.46,
        "lr_scheduler_patience": 7,
        "lr_scheduler_min_lr": 1e-05,
        "early_stopping_min_delta": 0.01,
        "gradient_clip_val": 1,

        "num_samples": 1,
        "mc_dropout": True,
    }
    return hyperparameters
