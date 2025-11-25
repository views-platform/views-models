
def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.
    This configuration is "operational" so modifying these settings will impact the model's behavior during the training.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for training the model, which determine the model's behavior during the training phase.
    """
    
    hyperparameters = {
        "steps": [*range(1, 36 + 1, 1)],
        "num_samples": 500,
        "mc_dropout": True,

        "batch_size": 64,
        "decoder_output_dim": 32,
        "delta": 0.1425562139624501,
        "dropout": 0.4,
        "early_stopping_min_delta": 0.005,
        "early_stopping_patience": 10,
        "false_negative_weight": 6.808821877873163,
        "false_positive_weight": 1.560537933024682,
        "feature_scaler": "RobustScaler",
        "force_reset": True,
        "gradient_clip_val": 0.91744867128012,
        "hidden_size": 64,
        "input_chunk_length": 36,
        "log_features": [
            "lr_ged_sb",
            "lr_ged_ns",
            "lr_ged_os",
            "lr_acled_sb",
            "lr_acled_os",
            "lr_ged_sb_tsum_24",
            "lr_splag_1_ged_sb",
            "lr_splag_1_ged_os",
            "lr_splag_1_ged_ns",
            "lr_wdi_sm_pop_netm",
            "lr_wdi_sm_pop_refg_or",
            "lr_wdi_sp_dyn_imrt_fe_in",
            "lr_wdi_ny_gdp_mktp_kd"
        ],
        "log_targets": True,
        "loss_function": "WeightedPenaltyHuberLoss",
        "lr": 0.00014739725450640124,
        "lr_scheduler_factor": 0.38192257067339064,
        "lr_scheduler_min_lr": 0.00001,
        "lr_scheduler_patience": 7,
        "n_epochs": 300,
        "non_zero_weight": 6.9673927758239085,
        "num_decoder_layers": 2,
        "num_encoder_layers": 1,
        "output_chunk_shift": 0,
        "random_state": 2023,
        "target_scaler": "RobustScaler",
        "temporal_decoder_hidden": 128,
        "temporal_hidden_size_future": None,
        "temporal_hidden_size_past": 16,
        "temporal_width_future": 6,
        "temporal_width_past": 4,
        "use_layer_norm": True,
        "use_static_covariates": False,
        "weight_decay": 0.0007039914716229751,
        "zero_threshold": 0.28929505832987634
    }


    return hyperparameters