
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
        "activation": "LeakyReLU",
        "batch_size": 128,
        "delta": 0.13666276790524423,
        "dropout": 0.2,
        "early_stopping_min_delta": 0.01,
        "early_stopping_patience": 10,
        "false_negative_weight": 5.639170553294556,
        "false_positive_weight": 2.9429523407286284,
        "feature_scaler": "RobustScaler",
        "force_reset": True,
        "gradient_clip_val": 0.5682200737952534,
        "hidden_dim": 64,
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
        "lr": 0.00000638766686260683,
        "lr_scheduler_factor": 0.28037295668072854,
        "lr_scheduler_min_lr": 0.000001,
        "lr_scheduler_patience": 5,
        "n_epochs": 300,
        "n_rnn_layers": 2,
        "non_zero_weight": 3.983113571716415,
        "output_chunk_shift": 1,
        "random_state": 123,
        "rnn_type": "GRU",
        "target_scaler": "RobustScaler",
        "use_reversible_instance_norm": True,
        "weight_decay": 0.00012486540017946828,
        "zero_threshold": 0.32936462133448563
    }


    return hyperparameters