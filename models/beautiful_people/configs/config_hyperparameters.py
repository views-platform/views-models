def get_hp_config():
    """
    N-HiTS hyperparameters from SpotlightLossLogcosh sweep best run.
    https://wandb.ai/views_pipeline/revolving_door_nhits_spotlight_v11_3_sweep/runs/p89rxmzk
    Returns:
    - hyperparameters (dict): Training configuration dictionary.
    """
    # r7
    hyperparameters = {
        # Temporal
        "steps": [*range(1, 36 + 1)],
        "input_chunk_length": 36,
        "output_chunk_length": 36,
        "output_chunk_shift": 0,
        "random_state": 67,
        "time_steps": 36,  # Checksum: Must match len(steps)

        # Inference
        "num_samples": 500,
        "mc_dropout": True,
        "n_jobs": -1,

        # Training
        "batch_size": 128,
        "n_epochs": 300,
        "early_stopping_patience": 8,
        "early_stopping_min_delta": 0.002,
        "force_reset": True,

        # Optimizer
        "optimizer_cls": "AdamW",
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "gradient_clip_val": 10.0,

        # LR Scheduler
        "lr_scheduler_cls": "ReduceLROnPlateau",
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 6,
        "lr_scheduler_min_lr": 3e-6,
        "lr_scheduler_kwargs": {
            "mode": "min",
            "factor": 0.5,
            "patience": 6,
            "min_lr": 3e-6,
            "cooldown": 0,
            "threshold": 0.002,
            "threshold_mode": "rel",
        },
        "early_stopping_monitor": "val_metrics/MSLE",
        "lr_scheduler_monitor": "val_metrics/MSLE",

        "optimizer_kwargs": {
            "betas": (0.9, 0.999), 
            "lr": 1e-4,
            "weight_decay": 1e-4,
        },

        # SpotlightLossLogcosh: logcosh base shape (gradient saturates at ±1)
        # Safe for basis-expansion architectures — bounded gradients prevent
        # learned interpolation coefficients from growing unbounded.
        "loss_function": "SpotlightLossLogcosh",
        "non_zero_threshold": 0.88,

        # Scaling
        "feature_scaler": None,
        "force_target_only": True,
        "target_scaler": "AsinhTransform",

        # N-HiTS Architecture
        "num_stacks": 2,
        "num_blocks": 1,
        "num_layers": 2,
        "layer_widths": 32,
        "pooling_kernel_sizes": [[4], [2]],
        "n_freq_downsample": [[3], [1]],
        "activation": "ReLU",
        "dropout": 0.3,
        "use_static_covariates": True,
        "use_reversible_instance_norm": True,
        "max_pool_1d": True,
        "checkpoint_mode": "best",
        # "static_covariate_stats": {
        #     "transform": "AsinhTransform",
        #     "inject": False,
        # },
        # Temporal Encodings
        # ModelCatalog reads this flag and injects the appropriate cyclic
        # encoder functions for the dataset temporal resolution, inferred
        # from config["level"] (e.g. cm→monthly, cd→daily, cw→weekly).
        "use_cyclic_encoders": False,
    }

    return hyperparameters