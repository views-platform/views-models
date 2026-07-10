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
        "num_samples": 1,
        "mc_dropout": False,
        "n_jobs": -1,

        # Training
        "batch_size": 128,
        "n_epochs": 300,
        "early_stopping_patience": 20,
        "early_stopping_min_delta": 0.001,
        "force_reset": True,

        # Optimizer
        "optimizer_cls": "AdamW",
        "lr": 1e-3,
        "weight_decay": 3e-4,
        "gradient_clip_val": 50.0,

        # LR Scheduler
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

        "optimizer_kwargs": {
            "lr": 1e-3,
            "weight_decay": 3e-4,
        },

        # SpotlightLossLogcosh: logcosh base shape (gradient saturates at ±1)
        # Safe for basis-expansion architectures — bounded gradients prevent
        # learned interpolation coefficients from growing unbounded.
        "loss_function": "SpotlightLossLogcosh",
        "non_zero_threshold": 0.88,
        "delta": 0.041685644972051974,

        # Scaling
        "feature_scaler": "AsinhTransform->MaxAbsScaler",  # global chain; covariates derive from the queryset (ADR-013, C-95) — reintroduce a feature_scaler_map group only when non-count features arrive
        "target_scaler": "AsinhTransform",

        # N-HiTS Architecture
        "num_stacks": 3,
        "num_blocks": 2,
        "num_layers": 3,
        "layer_widths": 256,
        "pooling_kernel_sizes": [[4, 4], [2, 2], [1, 1]],
        "n_freq_downsample": [[4, 4], [2, 2], [1, 1]],
        "activation": "Tanh",
        "dropout": 0.1,
        "use_static_covariates": True,
        "use_reversible_instance_norm": True,
        "max_pool_1d": True,
        "checkpoint_mode": "best",
        # "static_covariate_stats": {
        #     "transform": "AsinhTransform->MaxAbsScaler",
        #     "inject": False,
        # },
        # Temporal Encodings
        # ModelCatalog reads this flag and injects the appropriate cyclic
        # encoder functions for the dataset temporal resolution, inferred
        # from config["level"] (e.g. cm→monthly, cd→daily, cw→weekly).
        "use_cyclic_encoders": True,
    }

    return hyperparameters