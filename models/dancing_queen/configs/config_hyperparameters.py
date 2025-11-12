
def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.
    This configuration is "operational" so modifying these settings will impact the model's behavior during the training.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for training the model, which determine the model's behavior during the training phase.
    """
    
    # https://wandb.ai/views_pipeline/dancing_queen_rnn_sweep/runs/392wtfjg
    hyperparameters = {
        "steps": [*range(1, 36 + 1, 1)],
        "num_samples": 1,
        "mc_dropout": True,

        "activation": "ReLU",
        "batch_size": 128,
        "delta": 1.3845281556380034,
        "dropout": 0.4,
        "early_stopping_patience": 7,
        "false_negative_weight": 9.030870044206353,
        "false_positive_weight": 3.513477533071985,
        "feature_scaler": "MinMaxScaler",
        "gradient_clip_val": 1.2930308685191956,
        "hidden_dim": 128,
        "input_chunk_length": 60,
        "loss_function": "WeightedPenaltyHuberLoss",
        "lr": 0.00008240234281922644,
        "n_epochs": 300,
        "n_rnn_layers": 3,
        "non_zero_weight": 5.988359066258992,
        "rnn_type": "GRU",
        "target_scaler": "LogTransform",
        "use_reversible_instance_norm": True,
        "weight_decay": 0.00000697682612339767,
        "zero_threshold": 0.6179704191756129
    }


    return hyperparameters