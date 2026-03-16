
def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.
    This configuration is "operational" so modifying these settings will impact the model's behavior during the training.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for training the model, which determine the model's behavior during the training phase.
    """
    
    hyperparameters = {
        "steps": [*range(1, 36 + 1, 1)],
        "time_steps": 36,

        "num_samples": 1,
        "mc_dropout": True,

        'batch_size': 64,
        'decoder_output_dim': 32,
        'delta': 0.9867701013293054,
        'dropout': 0.3,
        'early_stopping_patience': 8,
        'false_negative_weight': 5.725193416698594,
        'false_positive_weight': 2.5971113722397408,
        'feature_scaler': 'MinMaxScaler',
        'gradient_clip_val': 0.55759113277096,
        'hidden_size': 512,
        'input_chunk_length': 72,
        'loss_function': 'WeightedPenaltyHuberLoss',
        'lr': 0.004591420805913535,
        'n_epochs': 300,
        'non_zero_weight': 6.181939476003923,
        'num_decoder_layers': 2,
        'num_encoder_layers': 2,
        'target_scaler': 'LogTransform',
        'temporal_decoder_hidden': 32,
        'temporal_width_future': 4,
        'temporal_width_past': 2,
        'use_layer_norm': True,
        'weight_decay': 0.00000253889071290023,
        'zero_threshold': 0.6323834242080557,

        "output_chunk_length": 36,
        "output_chunk_shift": 0,
        "random_state": 1,
        "optimizer_cls": "Adam",
        "lr_scheduler_factor": 0.46,
        "lr_scheduler_patience": 7,
        "lr_scheduler_min_lr": 1e-05,
        "early_stopping_min_delta": 0.01,
    }

    return hyperparameters