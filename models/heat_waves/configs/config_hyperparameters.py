from statistics import quantiles


def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.
    This configuration is "operational" so modifying these settings will impact the model's behavior during the training.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for training the model, which determine the model's behavior during the training phase.
    """

    hyperparameters = {
        # --- Forecast horizon ---
        "steps": [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
        ],
        # --- Input / output structure ---
        "input_chunk_length": 48,
        "output_chunk_length": 36,
        "output_chunk_shift": 0,
        # --- Architecture (TFT) ---
        "hidden_size": 128,
        "lstm_layers": 1,
        "num_attention_heads": 4,
        "full_attention": False,
        "feed_forward": "GELU",
        "hidden_continuous_size": 16,
        "add_relative_index": True,
        "norm_type": "LayerNorm",
        "use_static_covariates": True,
        "output_dim": [1, 1],
        "num_static_components": 1,
        # --- Regularization ---
        "dropout": 0.1,
        "weight_decay": 0.0001,
        "gradient_clip_val": 1,
        # --- Optimization ---
        "optimizer_cls": "Adam",
        "optimizer_kwargs": {
            "lr": 0.0004295014201718642,
            "weight_decay": 0.0001,
        },
        "lr": 0.0004295014201718642,
        # --- LR Scheduler ---
        "lr_scheduler_cls": "ReduceLROnPlateau",
        "lr_scheduler_factor": 0.46,
        "lr_scheduler_min_lr": 1e-05,
        "lr_scheduler_patience": 3,
        "lr_scheduler_kwargs": {
            "factor": 0.46,
            "min_lr": 1e-05,
            "mode": "min",
            "monitor": "train_loss",
            "patience": 3,
        },
        # --- Early Stopping ---
        "early_stopping_patience": 3,
        "early_stopping_min_delta": 0.01,
        # --- Loss ---
        "loss_function": "WeightedPenaltyHuberLoss",
        "delta": 0.025,
        "zero_threshold": 0.01,
        "non_zero_weight": 10,
        "false_positive_weight": 1,
        "false_negative_weight": 5,
        "likelihood": None,
        # --- Scaling & transforms ---
        "feature_scaler": "MinMaxScaler",
        "target_scaler": "RobustScaler",
        "log_targets": True,
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
            "lr_wdi_ny_gdp_mktp_kd",
        ],
        "use_reversible_instance_norm": False,
        "skip_interpolation": False,
        # --- Training ---
        "batch_size": 16,
        "n_epochs": 150,
        "force_reset": True,
        "random_state": 1,
        # --- Probabilistic / sampling ---
        "num_samples": 1,
        "mc_dropout": False,
    }

    return hyperparameters
