def get_hp_config():
    """
    N-BEATS hyperparameters
    """
    # r9
    hyperparameters = {
        # --- Forecast horizon ---
        "steps": list(range(1, 37)),

        # --- Architecture ---
        "generic_architecture": True,
        "num_stacks": 1,
        "num_blocks": 1,
        "num_layers": 2,
        "layer_widths": 16,
        "expansion_coefficient_dim": 16,
        "trend_polynomial_degree": 2,
        "activation": "GELU",
        "dropout": 0.3,
        "batch_norm": False,
        "use_reversible_instance_norm": True,
        "use_static_covariates": True,
        "use_cyclic_encoders": False,

        # --- Input / output structure ---
        "input_chunk_length": 36,
        "output_chunk_length": 36,
        "output_chunk_shift": 0,

        # --- Training ---
        "batch_size": 4096,
        "n_epochs": 300,
        "early_stopping_patience": 12,
        "early_stopping_min_delta": 0.002,
        "force_reset": True,

        # --- Optimizer ---
        "optimizer_cls": "AdamW",
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "gradient_clip_val": 1.0,
        "optimizer_kwargs": {
            "betas": (0.9, 0.999), 
            "lr": 1e-4,
            "weight_decay": 1e-4,
        },

        # --- LR Scheduler ---
        "lr_scheduler_cls": "ReduceLROnPlateau",
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 8,
        "lr_scheduler_min_lr": 3e-6,
        "lr_scheduler_kwargs": {
            "mode": "min",
            "factor": 0.5,
            "patience": 8,
            "min_lr": 3e-6,
            "cooldown": 0,
            "threshold": 0.002,
            "threshold_mode": "rel",
        },
        "early_stopping_monitor": "val_metrics/MSLE",
        "lr_scheduler_monitor": "val_metrics/MSLE",

        # --- Scaling ---
        "target_scaler": "AsinhTransform",
        "feature_scaler": None,
        "force_target_only": True,
    
        # --- Loss: SpotlightLoss v36 ---
        "loss_function": "MSE",
        "non_zero_threshold": 0.88,  # asinh(1) ≈ 0.88 in asinh space (1 battle death)
        "delta": 0.07139486580318413,

        # --- Prediction ---
        "likelihood": None,
        "num_samples": 1,
        "mc_dropout": False,

        # --- Other ---
        "random_state": 67,
        "time_steps": 36,  # Checksum: Must match len(steps)

        # --- other ---
        "n_jobs": -1
    }

    return hyperparameters