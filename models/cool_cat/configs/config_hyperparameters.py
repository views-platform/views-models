
def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.
    This configuration is "operational" so modifying these settings will impact the model's behavior during the training.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for training the model, which determine the model's behavior during the training phase.
    """
    
    hyperparameters = {
        # Temporal configuration
        "steps": [*range(1, 36 + 1, 1)],
        "input_chunk_length": 48,  # Longer context for conflict patterns
        "output_chunk_shift": 0,
        
        # Inference settings
        "num_samples": 500,
        "mc_dropout": True,

        # Training basics
        "batch_size": 64,
        "n_epochs": 300,
        "early_stopping_min_delta": 0.001,
        "early_stopping_patience": 12,
        "gradient_clip_val": 0.8,
        "force_reset": True,
        "random_state": 67,

        # Optimizer settings
        "lr": 1e-4,
        "weight_decay": 5e-4,
        "lr_scheduler_factor": 0.3,
        "lr_scheduler_min_lr": 1e-6,
        "lr_scheduler_patience": 5,

        # Scaling - AsinhTransform is optimal for zero-inflated targets
        "feature_scaler": "RobustScaler",
        "target_scaler": "AsinhTransform",  # Best for zero-inflated data
        "log_targets": False,  # AsinhTransform handles this
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

        # TiDE Architecture - optimized for conflict forecasting
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "decoder_output_dim": 32,
        "hidden_size": 128,
        "temporal_width_past": 4,
        "temporal_width_future": 4,
        "temporal_hidden_size_past": 32,
        "temporal_hidden_size_future": 32,
        "temporal_decoder_hidden": 64,
        "use_layer_norm": True,
        "dropout": 0.3,
        "use_static_covariates": False,  # Country/priogrid info helps
        "use_reversible_instance_norm": True,  # Critical for non-stationary conflict data

        # Loss function - ZeroInflatedLoss for sparse conflict data
        "loss_function": "ZeroInflatedLoss",
        "zero_weight": 1.0,        # Weight for zero/non-zero classification
        "count_weight": 2.0,       # Weight for intensity prediction
        "delta": 0.5,              # Huber delta for count component
        "zero_threshold": 0.01,    # Threshold for zero classification
    }


    return hyperparameters