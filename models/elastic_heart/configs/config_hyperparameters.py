
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
        
        # classic-sweep-120
        "activation": "Tanh",
        "batch_size": 32,
        "delta": 0.18801030586157305,
        "dropout": 0.3,
        "early_stopping_patience": 6,
        "false_negative_weight": 3.301437035350488,
        "false_positive_weight": 4.293752976960905,
        "feature_scaler": None,
        "ff_size": 64,
        "hidden_size": 128,
        "input_chunk_length": 36,
        # "loss_function": "WeightedPenaltyHuberLoss",
        "likelihood_function": "zinb_likelihood",
        "lr": 0.00004713247142351203,
        "n_epochs": 300,
        "non_zero_weight": 4.7850572475316415,
        "norm_type": "TimeBatchNorm2d",
        "normalize_before": False,
        "num_blocks": 3,
        "target_scaler": None,
        "weight_decay": 0.007733920154915341,
        "zero_threshold": 0.08236470534880172,

        "num_samples": 1,
        "mc_dropout": True,
    }
    return hyperparameters
