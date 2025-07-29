
def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.
    This configuration is "operational" so modifying these settings will impact the model's behavior during the training.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for training the model, which determine the model's behavior during the training phase.
    """
    
    # Works
    # hyperparameters = {
    #     "steps": [*range(1, 36 + 1, 1)],
    #     "feature_scaler": "StandardScaler",
    #     "target_scaler": "MaxAbsScaler",
    #     # 'StandardScaler','RobustScaler','MinMaxScaler','MaxAbsScaler'
    #     "temporal_decoder_hidden": 32,
    #     # model parameters
    #     "dropout": 0.2,
    #     "n_epochs": 150,
    #     "batch_size": 128,
    #     "delta": 0.2,
    #     "hidden_size": 512,
    #     "input_chunk_length": 48,
    #     "lr": 0.000040713,
    #     "num_samples": 1,
    #     "mc_dropout": True,
    #     "non_zero_weight": 12.0,
    #     "num_encoder_layers": 1,
    #     "early_stopping_patience": 15,
    #     "loss_function": "WeightedHuberLoss",
    # }


    # scarlet-sweep-7 - Works
    # hyperparameters = {
    #     "steps": [*range(1, 36 + 1, 1)],
        
    #     "batch_size": 64,
    #     "decoder_output_dim": 16,
    #     "delta": 0.8795196907285568,
    #     "dropout": 0.3,
    #     "early_stopping_patience": 2,
    #     "feature_scaler": "YeoJohnsonTransform",
    #     "gradient_clip_val": 0.8,
    #     "hidden_size": 1024,
    #     "input_chunk_length": 72,
    #     "loss_function": "WeightedHuberLoss",
    #     "lr": 0.0007712821660037065,
    #     "n_epochs": 120,
    #     "non_zero_weight": 13.253844420542856,
    #     "num_decoder_layers": 2,
    #     "num_encoder_layers": 2,
    #     "target_scaler": "RobustScaler",
    #     "temporal_decoder_hidden": 64,
    #     "use_layer_norm": False,
    #     "weight_decay": 0.00000245449547818616,
    #     "zero_threshold": 0.012980878408140745,

    #     "num_samples": 1,
    #     "mc_dropout": True,
    # }

    # celestial-sweep-115
    hyperparameters = {
        "steps": [*range(1, 36 + 1, 1)],
        
        "batch_size": 64,
        "decoder_output_dim": 8,
        "delta": 0.5057949004948398,
        "dropout": 0.4,
        "early_stopping_patience": 7,
        "feature_scaler": None,
        "gradient_clip_val": 1,
        "hidden_size": 128,
        "input_chunk_length": 24,
        "loss_function": "WeightedHuberLoss",
        "lr": 0.0002049641177218249,
        "n_epochs": 3,
        "non_zero_weight": 9.750135344178313,
        "num_decoder_layers": 2,
        "num_encoder_layers": 3,
        "target_scaler": "MinMaxScaler",
        "temporal_decoder_hidden": 64,
        "use_layer_norm": False,
        "weight_decay": 0.00000195227196867771,
        "zero_threshold": 0.009411321834189218,

        "num_samples": 75,
        "mc_dropout": True,
    }


    return hyperparameters

