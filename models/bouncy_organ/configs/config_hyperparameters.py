
def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.
    This configuration is "operational" so modifying these settings will impact the model's behavior during the training.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for training the model, which determine the model's behavior during the training phase.
    """
    
    hyperparameters = {
    
        # --- Forecast horizon ---
        "steps": list(range(1, 36 + 1)),
    
        # --- Sampling ---
        "num_samples": 1,
        "mc_dropout": False,
    
        # --- Architecture ---
        "activation": "ReLU",
        "hidden_size": 128,
        "ff_size": 128,
        "num_blocks": 3,
        "input_chunk_length": 24,
        "output_chunk_length": 36,
        "output_chunk_shift": 0,
        "norm_type": "LayerNorm",
        "normalize_before": False,
        "use_static_covariates": True,
    
        # --- Regularization ---
        "dropout": 0.3,
        "weight_decay": 0.0001,
        "gradient_clip_val": 1,
    
        # --- Optimization ---
        "optimizer_cls": "Adam",
        "lr": 0.00014162205399204303,
    
        # --- LR Scheduler ---
        "lr_scheduler_cls": "ReduceLROnPlateau",
        "lr_scheduler_factor": 0.46,
        "lr_scheduler_min_lr": 1e-05,
        "lr_scheduler_patience": 7,
    
        # --- Early Stopping ---
        "early_stopping_min_delta": 0.01,
        "early_stopping_patience": 15,
    
        # --- Loss ---
        "loss_function": "WeightedPenaltyHuberLoss",
        "delta": 0.025,
        "false_negative_weight": 5,
        "false_positive_weight": 1,
        "non_zero_weight": 5,
        "zero_threshold": 0.01,
    
        # --- Scaling ---
        "feature_scaler": "MinMaxScaler",
        "target_scaler": "MinMaxScaler",
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
    
        # --- Training ---
        "batch_size": 16,
        "n_epochs": 150,
        "force_reset": True,
        "use_reversible_instance_norm": False,
        "random_state": 1,
    }
    return hyperparameters
