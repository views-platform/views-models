
def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.
    This configuration is "operational" so modifying these settings will impact the model's behavior during the training.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for training the model, which determine the model's behavior during the training phase.
    """
    hyperparameters = {
        "steps": [*range(1, 36 + 1, 1)],

        # sweepy-sweep-138
        # "activation": "LeakyReLU",
        # "batch_size": 128,
        # "delta": 0.30039541729668506,
        # "dropout": 0.2,
        # "early_stopping_patience": 6,
        # "false_negative_weight": 14.111482163958568,
        # "false_positive_weight": 7.737139197766234,
        # "feature_scaler": "MinMaxScaler",
        # "generic_architecture": True,
        # "gradient_clip_val": 0.7,
        # "input_chunk_length": 36,
        # "layer_widths": 64,
        # "loss_function": "WeightedPenaltyHuberLoss",
        # "lr": 0.00000171647022363474,
        # "n_epochs": 300,
        # "non_zero_weight": 11.203128418130904,
        # "num_blocks": 4,
        # "num_stacks": 2,
        # "target_scaler": "MinMaxScaler",
        # "weight_decay": 0.0014995416335927957,
        # "zero_threshold": 0.19756112394407427,

        # giddy-sweep-1646
        # "activation": "Softplus",
        # "batch_size": 128,
        # "delta": 0.7785442989696792,
        # "dropout": 0.2,
        # "early_stopping_patience": 6,
        # "false_negative_weight": 9.029765081047252,
        # "false_positive_weight": 7.490488223499692,
        # "feature_scaler": "MinMaxScaler",
        # "generic_architecture": False,
        # "gradient_clip_val": 1,
        # "input_chunk_length": 36,
        # "layer_widths": 128,
        # "loss_function": "WeightedPenaltyHuberLoss",
        # "lr": 0.00000607338505399934,
        # "n_epochs": 300,
        # "non_zero_weight": 9.316795811463242,
        # "num_blocks": 3,
        # "num_stacks": 3,
        # "target_scaler": None,
        # "weight_decay": 0.0004370870656569707,
        # "zero_threshold": 0.2038769055426101,

        # eager-sweep-1524
        "activation": "GELU",
        "batch_size": 32,
        "delta": 4.490477768407159,
        "dropout": 0.5,
        "early_stopping_patience": 6,
        "false_negative_weight": 3.8454464749871344,
        "false_positive_weight": 3.6875791242302265,
        "feature_scaler": "MaxAbsScaler",
        "generic_architecture": False,
        "gradient_clip_val": 0.7,
        "input_chunk_length": 24,
        "layer_widths": 128,
        "loss_function": "WeightedPenaltyHuberLoss",
        "lr": 0.00013236695574709746,
        "n_epochs": 300,
        "non_zero_weight": 7.4990355785276765,
        "num_blocks": 1,
        "num_stacks": 4,
        "target_scaler": "MinMaxScaler",
        "weight_decay": 0.0007710831110161143,
        "zero_threshold": 0.2871542855875679,

        "num_samples": 1,
        "mc_dropout": True,
    }
    return hyperparameters

