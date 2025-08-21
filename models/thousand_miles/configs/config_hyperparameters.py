
def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.
    This configuration is "operational" so modifying these settings will impact the model's behavior during the training.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for training the model, which determine the model's behavior during the training phase.
    """
    
    hyperparameters = {
        "steps": [*range(1, 36 + 1, 1)],
        # Good! silvery-sweep-1
        # "batch_size": 64,
        # "decoder_output_dim": 8,
        # "delta": 3.696974087786076,
        # "dropout": 0.2,
        # "early_stopping_patience": 2,
        # "false_negative_weight": 5.0977656184083395,
        # "false_positive_weight": 4.507821247953932,
        # "feature_scaler": None,
        # "gradient_clip_val": 0.2,
        # "hidden_size": 256,
        # "input_chunk_length": 72,
        # "loss_function": "WeightedPenaltyHuberLoss",
        # "lr": 0.00009976652946422868,
        # "n_epochs": 5,
        # "non_zero_weight": 1.7773296987133582,
        # "num_decoder_layers": 4,
        # "num_encoder_layers": 1,
        # "target_scaler": None,
        # "temporal_decoder_hidden": 32,
        # "use_layer_norm": True,
        # "weight_decay": 0.00533531735917749,
        # "zero_threshold": 0.17066172076836716,

        #crimson-sweep-169. Psychotic.
        # "batch_size": 128,
        # "decoder_output_dim": 16,
        # "delta": 1.7234705394805443,
        # "dropout": 0.1,
        # "early_stopping_patience": 6,
        # "false_negative_weight": 9.3313776622516,
        # "false_positive_weight": 10.78492842081859,
        # "feature_scaler": "MaxAbsScaler",
        # "gradient_clip_val": 0.8,
        # "hidden_size": 128,
        # "input_chunk_length": 36,
        # "loss_function": "WeightedPenaltyHuberLoss",
        # "lr": 0.00001471988186818606,
        # "n_epochs": 300,
        # "non_zero_weight": 13.23535161231906,
        # "num_decoder_layers": 2,
        # "num_encoder_layers": 5,
        # "target_scaler": "MaxAbsScaler",
        # "temporal_decoder_hidden": 16,
        # "use_layer_norm": False,
        # "weight_decay": 0.0000214973309074918,
        # "zero_threshold": 0.26369001329592373,

        #peachy-sweep-64
        "batch_size": 32,
        "decoder_output_dim": 32,
        "delta": 3.402036667021411,
        "dropout": 0.4,
        "early_stopping_patience": 6,
        "false_negative_weight": 0.4094813655878722,
        "false_positive_weight": 5.344567094760235,
        "feature_scaler": None,
        "gradient_clip_val": 0.2,
        "hidden_size": 32,
        "input_chunk_length": 48,
        "loss_function": "WeightedPenaltyHuberLoss",
        "lr": 0.0008657902172163073,
        "n_epochs": 300,
        "non_zero_weight": 13.833818931287857,
        "num_decoder_layers": 2,
        "num_encoder_layers": 2,
        "target_scaler": "MinMaxScaler",
        "temporal_decoder_hidden": 16,
        "use_layer_norm": False,
        "weight_decay": 0.00010196612489718342,
        "zero_threshold": 0.0965370826949934,


        "num_samples": 1,
        "mc_dropout": True,
    }


    return hyperparameters