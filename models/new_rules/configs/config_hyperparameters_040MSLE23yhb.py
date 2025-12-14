def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.

    NOTE: This configuration was restored from a previous best-performing run
    (MSLE ~0.40 from wandb run `3j9x0s4r`) to establish a replicable baseline.
    """

    hyperparameters = {
        # --- From Best Old Model ---
        "steps": [*range(1, 36 + 1, 1)],
        "activation": "LeakyReLU",
        "generic_architecture": True,
        "num_stacks": 1,
        "num_blocks": 1,
        "num_layers": 1,
        "layer_widths": 64,
        "dropout": 0.3,
        "input_chunk_length": 24,
        "output_chunk_shift": 0,
        "batch_size": 8,
        "n_epochs": 300,
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 0.001,
        "lr": 0.0005873328851386325,
        "weight_decay": 0.0003292268280079564,
        "lr_scheduler_factor": 0.46300979785707297,
        "lr_scheduler_min_lr": 0.00001,
        "lr_scheduler_patience": 7,
        "gradient_clip_val": 0.6336557913524701,
        "feature_scaler": "MinMaxScaler",
        "target_scaler": "MinMaxScaler",
        "log_targets": True,
        "log_features": [
            "lr_ged_sb", "lr_ged_ns", "lr_ged_os",
            "lr_acled_sb", "lr_acled_os",
            "lr_ged_sb_tsum_24",
            "lr_splag_1_ged_sb", "lr_splag_1_ged_os", "lr_splag_1_ged_ns",
            "lr_wdi_sm_pop_netm", "lr_wdi_sm_pop_refg_or",
            "lr_wdi_sp_dyn_imrt_fe_in", "lr_wdi_ny_gdp_mktp_kd",
        ],
        "loss_function": "WeightedPenaltyHuberLoss",
        "delta": 0.129050050430042,
        "zero_threshold": 0.12953171739852642,
        "false_positive_weight": 1.4269851202559674,
        "false_negative_weight": 3.8819100926929138,
        "non_zero_weight": 2.504275866632825,
        "force_reset": True,
        "num_samples": 1,
        "mc_dropout": True,
    }

    return hyperparameters
