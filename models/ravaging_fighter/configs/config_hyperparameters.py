def get_hp_config():
    """
    N-BEATS hyperparameters
    """
    # r8
    hyperparameters = {
        # --- Forecast horizon ---
        "steps": list(range(1, 37)),

        # --- Architecture ---
        "generic_architecture": True,
        "num_stacks": 2,
        "num_blocks": 2,
        "num_layers": 3,
        "layer_widths": 256,
        "expansion_coefficient_dim": 512,
        "trend_polynomial_degree": 2,
        "activation": "GELU",
        "dropout": 0.1,
        "batch_norm": False,
        "use_reversible_instance_norm": True,
        "use_static_covariates": True,
        "use_cyclic_encoders": True,

        # --- Input / output structure ---
        "input_chunk_length": 36,
        "output_chunk_length": 36,
        "output_chunk_shift": 0,

        # --- Training ---
        "batch_size": 128,
        "n_epochs": 300,
        "early_stopping_patience": 20,
        "early_stopping_min_delta": 0.001,
        "force_reset": True,

        # --- Optimizer ---
        "optimizer_cls": "AdamW",
        "lr": 1e-3,
        "weight_decay": 3e-4,
        "gradient_clip_val": 50.0,
        "optimizer_kwargs": {
            "lr": 1e-3,
            "weight_decay": 3e-4,
        },

        # --- LR Scheduler ---
        "lr_scheduler_cls": "ReduceLROnPlateau",
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 10,
        "lr_scheduler_min_lr": 1e-6,
        "lr_scheduler_kwargs": {
            "mode": "min",
            "factor": 0.5,
            "patience": 10,
            "min_lr": 1e-6,
            "cooldown": 2,
            "threshold": 0.01,
            "threshold_mode": "rel",
        },

        # --- Scaling ---
        "target_scaler": "AsinhTransform",
        "feature_scaler": "AsinhTransform->MaxAbsScaler",  # global chain; covariates derive from the queryset (ADR-013, C-95) — reintroduce a feature_scaler_map group only when non-count features arrive

        # --- Loss: SpotlightLoss v36 ---
        "loss_function": "SpotlightLossLogcosh",
        "non_zero_threshold": 0.88,  # asinh(1) ≈ 0.88 in asinh space (1 battle death)
        "delta": 0.07139486580318413,

        # --- Prediction ---
        "likelihood": None,
        "num_samples": 1,
        "mc_dropout": False,

        # --- Other ---
        "random_state": 67,
        "time_steps": 36,  # Checksum: Must match len(steps)
        # "static_covariate_stats": {"transform": "AsinhTransform"},

        # --- other ---
        "n_jobs": -1
    }

    return hyperparameters