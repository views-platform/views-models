
def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.
    This configuration is "operational" so modifying these settings will impact the model's behavior during the training.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for training the model, which determine the model's behavior during the training phase.
    """
    
    hyperparameters = {
        'steps': [*range(1, 36 + 1, 1)],

        # "batch_size": 1024,
        # "delta": 4.296149099765397,
        # "dilation_base": 4,
        # "dropout": 0.4,
        # "early_stopping_patience": 6,
        # "false_negative_weight": 6.636201942047831,
        # "false_positive_weight": 9.75477950433154,
        # "feature_scaler": "MinMaxScaler",
        # "gradient_clip_val": 0.2,
        # "input_chunk_length": 60,
        # "kernel_size": 2,
        # "loss_function": "WeightedPenaltyHuberLoss",
        # "lr": 0.00000156636499165135,
        # "n_epochs": 300,
        # "non_zero_weight": 3.211420217955327,
        # "num_filters": 4,
        # "target_scaler": None,
        # "use_reversible_instance_norm": True,
        # "weight_decay": 0.009257262851648228,
        # "weight_norm": False,
        # "zero_threshold": 0.2668203820837245,

        # rare-sweep-27
        "batch_size": 1024,
        "delta": 2.6838245979469986,
        "dilation_base": 3,
        "dropout": 0.1,
        "early_stopping_patience": 6,
        "false_negative_weight": 11.396522481793252,
        "false_positive_weight": 12.338182270300305,
        "feature_scaler": "MinMaxScaler",
        "gradient_clip_val": 1,
        "input_chunk_length": 60,
        "output_chunk_length": 36,
        "output_chunk_shift": 0,
        "kernel_size": 6,
        "loss_function": "WeightedPenaltyHuberLoss",
        "lr": 0.0006380485804969655,
        "n_epochs": 300,
        "non_zero_weight": 12.28959993228442,
        "num_filters": 3,
        "target_scaler": "MaxAbsScaler",
        "use_reversible_instance_norm": True,
        "weight_decay": 0.00003613582849676987,
        "weight_norm": False,
        "zero_threshold": 0.05638024262912267,

        "num_samples": 1,
        "mc_dropout": False,
        "random_state": 1,
        'optimizer_cls': "Adam",
        'lr_scheduler_factor': 0.1,
        'lr_scheduler_patience': 6,
        'lr_scheduler_min_lr': 1e-5,
        'early_stopping_min_delta': 0.001,


    }
    return hyperparameters
