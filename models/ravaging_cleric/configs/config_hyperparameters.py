
def get_hp_config():
    """
    TSMixer hyperparameters
    Ported from tuning_202606 post-r8 ("fix elastic heart", 2026-06):
    lr=3e-4, clip=20, dropout=0.4, hidden=128, es_patience=25, RevIN=True
    """
    hyperparameters = {
        # Temporal
        "steps": [*range(1, 36 + 1, 1)],
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
        "early_stopping_patience": 25,
        "early_stopping_min_delta": 0.001,
        "force_reset": True,

        # Optimizer
        "optimizer_cls": "AdamW",
        "lr": 3e-4,
        "weight_decay": 3e-4,
        "gradient_clip_val": 20.0,

        # LR Scheduler
        "lr_scheduler_cls": "ReduceLROnPlateau",
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 15,
        "lr_scheduler_min_lr": 1e-6,
        "lr_scheduler_kwargs": {
            "mode": "min",
            "factor": 0.5,
            "patience": 15,
            "min_lr": 1e-6,
            "cooldown": 4,
            "threshold": 0.01,
            "threshold_mode": "rel",
        },
        "optimizer_kwargs": {
            "lr": 3e-4,
            "weight_decay": 3e-4,
        },
        "checkpoint_mode": "best",
        "loss_function": "SpotlightLossLogcosh",
        "non_zero_threshold": 0.88,

        # Scaling
        "feature_scaler": "AsinhTransform->MaxAbsScaler",  # global chain; covariates derive from the queryset (ADR-013, C-95) — reintroduce a feature_scaler_map group only when non-count features arrive
        "target_scaler": "AsinhTransform",

        # TSMixer Architecture
        "num_blocks": 3,
        "hidden_size": 128,
        "ff_size": 256,
        "activation": "GELU",
        "norm_type": "LayerNorm",
        "normalize_before": True,
        "dropout": 0.4,
        "use_static_covariates": True,
        "use_reversible_instance_norm": True,

        # "static_covariate_stats": {
        #     "transform": "AsinhTransform->MaxAbsScaler",
        #     "inject": True,
        #     # "stats": ["trend", "sparsity"],
        # },

        "use_cyclic_encoders": True,
    }
    return hyperparameters
