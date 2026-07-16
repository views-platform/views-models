def get_hp_config():
    """
    https://wandb.ai/views_pipeline/smol_cat_tide_shadow_20260505_A_sweep/runs/aaxcc2fh
    """
    
    hyperparameters = {
        "steps": [*range(1, 36 + 1, 1)],
        "time_steps": 36,  # Checksum: Must match len(steps)
        "n_jobs": -1,

        # TiDE Architecture
        "input_chunk_length": 36,
        "output_chunk_length": 36,
        "output_chunk_shift": 0,
        "hidden_size": 256,
        "decoder_output_dim": 64,
        "temporal_decoder_hidden": 96,
        "temporal_width_past": 32,
        # Future covariates are 3-wide; avoid unnecessary feature expansion noise.
        "temporal_width_future": 3,
        "temporal_hidden_size_past": 96,
        "temporal_hidden_size_future": 16,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "use_layer_norm": True,
        "use_reversible_instance_norm": True,
        "dropout": 0.3,
        "use_static_covariates": True,

        # Training
        "n_epochs": 300,
        "batch_size": 1024,
        "random_state": 67,
        "force_reset": True,

        # Optimizer
        "optimizer_cls": "AdamW",
        "lr": 2e-4,
        "weight_decay": 5e-5,
        "optimizer_kwargs": {
            "lr": 2e-4,
            "weight_decay": 5e-5,
        },

        # LR Scheduler
        "lr_scheduler_cls": "ReduceLROnPlateau",
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 5,
        "lr_scheduler_min_lr": 3e-6,
        "lr_scheduler_kwargs": {
            "mode": "min",
            "factor": 0.5,
            "patience": 5,
            "min_lr": 3e-6,
            "cooldown": 0,
            "threshold": 0.002,
            "threshold_mode": "rel",
        },
        "early_stopping_monitor": "val_metrics/MSLE",
        "lr_scheduler_monitor": "val_metrics/MSLE",
        # Trainer
        "gradient_clip_val": 25.0,
        "early_stopping_patience": 12,
        "early_stopping_min_delta": 0.002,

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
        "feature_scaler": None,
        "feature_scaler_map": {
            "AsinhTransform->MaxAbsScaler": [
                # Primary joint target variables
                # "lr_ged_sb",
                # "lr_ged_os",
                # "lr_ged_ns",

                # Natural and Social Geography features
                "lr_imr_mean",
                "lr_mountains_mean",
                "lr_dist_diamsec",
                "lr_dist_petroleum",
                "lr_agri_ih",
                "lr_barren_ih",
                "lr_forest_ih",
                "lr_pasture_ih",
                "lr_savanna_ih",
                "lr_shrub_ih",
                "lr_urban_ih",
                "ln_pop_gpw_sum",
                "ln_ttime_mean",
                "ln_gcp_mer",
                "ln_bdist3",
                "ln_capdist",
                "lr_greq_1_excluded",

                # Conflict decay memory features (mix of decay 12 and 24)
                "lr_decay_ged_sb_1",
                "lr_decay_ged_sb_5",
                "lr_decay_ged_sb_25",
                "lr_decay_ged_sb_100",
                "lr_decay_ged_sb_500",
                "lr_decay_ged_os_1",
                "lr_decay_ged_os_5",
                "lr_decay_ged_os_25",
                "lr_decay_ged_os_100",
                "lr_decay_ged_os_500",
                "lr_decay_ged_ns_5",
                "lr_decay_ged_ns_1",
                "lr_decay_ged_ns_25",
                "lr_decay_ged_ns_100",
                "lr_decay_ged_ns_500",

                # Spatial-temporal lag features
                "lr_splag_1_1_sb_1",
                "lr_splag_1_decay_ged_sb_1",
                "lr_splag_1_decay_ged_os_1",
                "lr_splag_1_decay_ged_ns_1",

                # Graph/tree and space-time spillover features
                "lr_treelag_1_sb",
                "lr_treelag_2_sb",
                "lr_treelag_1_os",
                "lr_treelag_2_os",
                "lr_treelag_1_ns",
                "lr_treelag_2_ns",
                "lr_sptime_dist_k1_ged_sb",
                "lr_sptime_dist_k10_ged_sb",
                "lr_sptime_dist_k001_ged_sb",
                "lr_sptime_dist_k1_ged_os",
                "lr_sptime_dist_k10_ged_os",
                "lr_sptime_dist_k001_ged_os",
                "lr_sptime_dist_k1_ged_ns",
                "lr_sptime_dist_k10_ged_ns",
                "lr_sptime_dist_k001_ged_ns",
            ],
        },

        # Encoders
        "use_cyclic_encoders": True,
        # "static_covariate_stats": {"transform": "AsinhTransform", "inject": True},
    }

    return hyperparameters
