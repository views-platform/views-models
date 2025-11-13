
def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.
    This configuration is "operational" so modifying these settings will impact the model's behavior during the training.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for training the model, which determine the model's behavior during the training phase.
    """
    
    hyperparameters = {
        "steps": [*range(1, 36 + 1, 1)],
        "num_samples": 1,
        "mc_dropout": True,
        "log_features": ["lr_ged_sb", "lr_ged_ns", "lr_ged_os", "lr_acled_sb", "lr_acled_os", "lr_ged_sb_tsum_24", 
                         "lr_splag_1_ged_sb", "lr_splag_1_ged_os", "lr_splag_1_ged_ns"],
        "log_targets": True,

        "activation": "ReLU",
        "batch_size": 256,
        "delta": 4.451296381697266,
        "dropout": 0.4,
        "early_stopping_patience": 7,
        "false_negative_weight": 7.40732576993341,
        "false_positive_weight": 4.85513573778521,
        "feature_scaler": "RobustScaler",
        "gradient_clip_val": 1.2779580850008343,
        "hidden_dim": 32,
        "input_chunk_length": 48,
        "loss_function": "WeightedPenaltyHuberLoss",
        "lr": 0.00020844667233098872,
        "n_epochs": 300,
        "n_rnn_layers": 1,
        "non_zero_weight": 3.488375573475057,
        "rnn_type": "GRU",
        "target_scaler": "RobustScaler",
        "use_reversible_instance_norm": True,
        "weight_decay": 0.0008697746930152339,
        "zero_threshold": 0.0005004157517819605
    }


    return hyperparameters