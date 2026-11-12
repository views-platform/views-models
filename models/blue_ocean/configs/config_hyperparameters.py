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
        "force_target_only": False,
                "feature_scaler_map": {
                    "AsinhTransform": [
                    "lr_ged_sb",
                    "lr_ged_ns",
                    "lr_ged_os",
                    "lr_agri_gc",
                    "lr_acled_battles",
                    "lr_acled_explosions",
                    "lr_acled_vac",
                    "lr_acled_protests",
                    "lr_acled_riots",
                    "lr_acled_strategic",
                    "lr_acled_fatalities",
                    "lr_aquaveg_gc",
                    "lr_barren_gc",
                    "lr_cmr_max",
                    "lr_cmr_mean",
                    "lr_cmr_min",
                    "lr_cmr_sd",
                    "lr_diamprim_s",
                    "lr_diamsec_s",
                    "lr_forest_gc",
                    "lr_gem_s",
                    "lr_goldplacer_s",
                    "lr_goldsurface_s",
                    "lr_goldvein_s",
                    "lr_growend",
                    "lr_growstart",
                    "lr_harvarea",
                    "lr_herb_gc",
                    "lr_ghspop_pop_count",
                    "lr_ghsbuilts_built_area",
                    "lr_imr_max",
                    "lr_imr_mean",
                    "lr_imr_min",
                    "lr_imr_sd",
                    "lr_landarea",
                    "lr_maincrop",
                    "lr_mountains_mean",
                    "lr_petroleum_s",
                    "lr_rainseas",
                    "lr_shdi_shdi",
                    "lr_shdi_healthindex",
                    "lr_shdi_edindex",
                    "lr_shdi_incindex",
                    "lr_shrub_gc",
                    "lr_ttime_max",
                    "lr_ttime_mean",
                    "lr_ttime_min",
                    "lr_ttime_sd",
                    "lr_urban_gc",
                    "lr_vdem_v2xcl_dmove",
                    "lr_vdem_v2xeg_eqdr",
                    "lr_vdem_v2xpe_exlsocgr",
                    "lr_vdem_v2x_clphy",
                    "lr_vdem_v2xcl_prpty",
                    "lr_vdem_v2x_ex_military",
                    "lr_vdem_v2x_ex_party",
                    "lr_vdem_v2x_horacc",
                    "lr_vdem_v2xnp_client",
                    "lr_vdem_v2xnp_regcorr",
                    "lr_vdem_v2xpe_exlgeo",
                    "lr_vdem_v2x_veracc",
                    "lr_vdem_v2xpe_exlpol",
                    "lr_vdem_v2x_diagacc",
                    "lr_vdem_v2x_divparctrl",
                    "lr_vdem_v2xeg_eqprotec",
                    "lr_vdem_v2x_genpp",
                    "lr_vdem_v2xpe_exlgender",
                    "lr_vdem_v2x_hosabort",
                    "lr_vdem_v2x_libdem",
                    "lr_vdem_v2xcl_rol",
                    "lr_vdem_v2x_accountability",
                    "lr_water_gc",
                    ],
                },
        
    
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

        # --- other ---
        "n_jobs": -1
    }

    return hyperparameters