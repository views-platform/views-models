def get_hp_config():
    """
    https://wandb.ai/views_pipeline/smol_cat_tide_shadow_20260505_A_sweep/runs/aaxcc2fh
    """
    
    hyperparameters = {
        # Steps
        "steps": [*range(1, 36 + 1, 1)],
        "time_steps": 36,  # Checksum: Must match len(steps)
        "n_jobs": -1,

        # TiDE Architecture
        "input_chunk_length": 36,
        "output_chunk_length": 36,
        "output_chunk_shift": 0,
        "hidden_size": 384,
        "decoder_output_dim": 64,
        "temporal_decoder_hidden": 128,
        "temporal_width_past": 24,
        "temporal_width_future": 4,
        "temporal_hidden_size_past": 128,
        "temporal_hidden_size_future": 32,
        "num_encoder_layers": 3,
        "num_decoder_layers": 2,
        "use_layer_norm": True,
        "use_reversible_instance_norm": True,
        "dropout": 0.1,
        "use_static_covariates": True,

        # Training
        "n_epochs": 300,
        "batch_size": 128,
        "random_state": 67,
        "force_reset": True,

        # Optimizer
        "optimizer_cls": "AdamW",
        "lr": 0.0005,
        "weight_decay": 0.0,
        "optimizer_kwargs": {
            "lr": 0.0005,
            "weight_decay": 0.0,
        },

        # LR Scheduler
        "lr_scheduler_cls": "ReduceLROnPlateau",
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 20,
        "lr_scheduler_min_lr": 1e-6,
        "lr_scheduler_kwargs": {
            "mode": "min",
            "factor": 0.5,
            "patience": 20,
            "min_lr": 1e-5,
            "cooldown": 5,
            "threshold": 0.01,
            "threshold_mode": "rel",
        },

        # Trainer
        "gradient_clip_val": 200,
        "early_stopping_patience": 35,
        "early_stopping_min_delta": 0.001,

        # Loss
        # "loss_function": "SpotlightLossLogcosh",
        "loss_function": "SpotlightLossLogcosh",
        #"delta": 0.06276537091497503,
        "non_zero_threshold": 0.88,

        # Prediction
        "likelihood": None,
        "num_samples": 1,
        "mc_dropout": False,

        # Scalers
        "target_scaler": "AsinhTransform",
        "feature_scaler": "AsinhTransform->MaxAbsScaler",  # global chain; covariates derive from the queryset (ADR-013, C-95) — reintroduce a feature_scaler_map group only when non-count features arrive

        # Encoders
        "use_cyclic_encoders": True,
        # "static_covariate_stats": {"transform": "AsinhTransform", "inject": True},
    }

    return hyperparameters
