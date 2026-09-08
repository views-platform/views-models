def get_hp_config():
    """
    N-HiTS hyperparameters for GDP per capita temporal disaggregation.

    Task: Disaggregate yearly GDP per capita (from country_year) into monthly
    estimates using monthly conflict covariates (ged_sb, ged_os, ged_ns),
    yearly population, and V-Dem democracy index.

    Architecture rationale:
    - 3 stacks for hierarchical multi-scale decomposition:
      Stack 1 (annual):    pooling=12, downsample=12 → 3 basis functions (yearly trend)
      Stack 2 (quarterly): pooling=4,  downsample=4  → 9 basis functions (seasonal pattern)
      Stack 3 (monthly):   pooling=1,  downsample=1  → 36 basis functions (fine detail)
    - AvgPool (max_pool_1d=False): GDP is smooth, no spikes to preserve
    - RevIN=True: normalizes cross-country GDP scale differences (~$250-$80,000)
    - SpotlightLoss with alpha=0, beta=0: reduces to Huber + temporal gradient Huber
      gamma > 0 is essential with RevIN to prevent flat mean-collapse predictions

    Returns:
    - hyperparameters (dict): Training configuration dictionary.
    """

    hyperparameters = {
        # Temporal
        "steps": [*range(1, 36 + 1, 1)],
        "input_chunk_length": 36,
        "output_chunk_length": 36,
        "output_chunk_shift": 0,
        "random_state": 67,
        # Inference
        "num_samples": 1,
        "mc_dropout": False,
        "n_jobs": -1,
        # Training
        "batch_size": 128,
        "n_epochs": 100,
        "early_stopping_patience": 4,
        "early_stopping_min_delta": 0.0005,
        "early_stopping_monitor": "val_metrics/MSLE",
        "checkpoint_monitor": "val_metrics/MSLE",
        "force_reset": True,
        # Optimizer
        "optimizer_cls": "AdamW",
        "lr": 0.0005,
        "weight_decay": 0.000001,
        "gradient_clip_val": 1.0,
        # LR Scheduler
        "lr_scheduler_cls": "ReduceLROnPlateau",
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 2,
        "lr_scheduler_min_lr": 3e-6,
        "lr_scheduler_monitor": "val_metrics/MSLE",
        "lr_scheduler_kwargs": {
            "mode": "min",
            "factor": 0.5,
            "patience": 2,
            "min_lr": 3e-6,
            "cooldown": 0,
            "threshold": 0.0005,
            "threshold_mode": "rel",
        },
        "optimizer_kwargs": {
            "lr": 0.0005,
            "weight_decay": 0.000001,
        },
        # Loss: SpotlightLoss (alpha=0, beta=0 → Huber + temporal gradient Huber)
        "loss_function": "SpotlightLossLogcosh",
        "alpha": 0.0,  # no magnitude weighting (GDP is always large in asinh space)
        "beta": 0.0,  # no asymmetry (over/under-prediction equally bad)
        "kappa": 0.0,  # unused when beta=0, but required by constructor
        # "delta": 10.0,  # Huber threshold
        "gamma": 0.35,  # temporal gradient penalty — prevents flat predictions with RevIN
        "delta": 0.0,  # Huber threshold — unused when beta=0, but required by constructor
        "non_zero_threshold": 0.88,
        # Scaling
        "feature_scaler": None,
        "target_scaler": "AsinhTransform",
        "feature_scaler_map": {
            "AsinhTransform": [
                "lr_pop_totl",
            ]
        },
        # N-HiTS Architecture: minimal 2-stack
        "num_stacks": 2,
        "num_blocks": 1,
        "num_layers": 2,
        "layer_widths": 64,  # 256 → 64: GDP needs minimal capacity
        "pooling_kernel_sizes": [[12], [1]],  # annual trend + monthly detail
        "n_freq_downsample": [[12], [1]],  # 3 + 36 basis functions
        "max_pool_1d": False,
        "activation": "GELU",
        "dropout": 0.3,  # 0.10 → 0.0: tiny model, no overfitting risk
        "use_reversible_instance_norm": True,
        "temporal_disaggregation": {
            "lr_gdp_pcap": {
                "method": "denton-cholette",
                "conversion": "average",
            },
            "lr_ttns_zs": {
                "method": "denton-cholette",
                "conversion": "average",
            },
            "lr_imrt_in": {
                "method": "denton-cholette",
                "conversion": "average",
            },
            "lr_le00_in": {
                "method": "denton-cholette",
                "conversion": "average",
            },
            "lr_chex_gd_zs": {  
                "method": "denton-cholette",
                "conversion": "average",
            },
            "lr_chex_gd_pcap": {
                "method": "denton-cholette",
                "conversion": "average",
            },
            "lr_stnt_me_zs": {
                "method": "denton-cholette",
                "conversion": "average",
            },
            "lr_defc_zs": {
                "method": "denton-cholette",
                "conversion": "average",
            },
            "lr_prm_enrr": {
                "method": "denton-cholette",
                "conversion": "average",
            },
            "lr_v2x_libdem": {
                "method": "denton-cholette",
                "conversion": "average",
            },
        },
    }

    return hyperparameters




