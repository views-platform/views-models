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
        "activation": "ReLU",
        "batch_size": 128,
        "delta": 0.1031278505804136,
        "dropout": 0.3,
        "early_stopping_min_delta": 0.01,
        "early_stopping_patience": 10,
        "false_negative_weight": 2.228018327985685,
        "false_positive_weight": 1.5395763736325097,
        "feature_scaler": "MinMaxScaler",
        "force_reset": True,
        "gradient_clip_val": 0.5415811286559888,
        "hidden_dim": 512,
        "input_chunk_length": 36,
        "output_chunk_length": 36,
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
        "lr": 0.00019633444110428468,
        "lr_scheduler_factor": 0.16611545771941605,
        "lr_scheduler_min_lr": 0.00001,
        "lr_scheduler_patience": 3,
        "n_epochs": 300,
        "n_rnn_layers": 3,
        "non_zero_weight": 3.9516967488800447,
        "output_chunk_shift": 0,
        "rnn_type": "GRU",
        "target_scaler": "RobustScaler",
        "use_reversible_instance_norm": False,
        "weight_decay": 0.0007293652167062485,
        "zero_threshold": 0.15954413640606746,
        "random_state": 1,
        "optimizer_cls": "Adam",
    }

    return hyperparameters
