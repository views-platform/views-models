
def get_hp_config():
    """
    TSMixer hyperparameters
    Ported from tuning_202606 post-r8 ("fix elastic heart", 2026-06):
    lr=3e-4, clip=20, dropout=0.4, hidden=128, es_patience=25, RevIN=True
    """
    # r8
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
        "batch_size": 4096,
        "n_epochs": 300,
        "early_stopping_monitor": "val_metrics/MSLE",
        "lr_scheduler_monitor": "val_metrics/MSLE",
        "early_stopping_patience": 8,
        "early_stopping_min_delta": 0.0003,
        "force_reset": True,

        # Optimizer
        "optimizer_cls": "AdamW",
        "lr": 0.0001,
        "weight_decay": 0.01,
        "gradient_clip_val": 1.0,
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
            "threshold": 0.0003,
            "threshold_mode": "rel",
        },
        "optimizer_kwargs": {
            "betas": (0.9, 0.999), 
            "lr": 0.0001,
            "weight_decay": 0.01,
        },
        "checkpoint_mode": "best",
        "loss_function": "SpotlightLossLogcosh",
        "non_zero_threshold": 0.88,

        # Scaling
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
        

        # TSMixer Architecture
        "num_blocks": 2,
        "hidden_size": 64,
        "ff_size": 128,
        "activation": "ReLU",
        "norm_type": "LayerNorm",
        "normalize_before": False,
        "dropout": 0.5,
        "use_static_covariates": True,
        "use_reversible_instance_norm": True,

        # "static_covariate_stats": {
        #     "transform": "AsinhTransform",
        #     "inject": True,
        #     # "stats": ["trend", "sparsity"],
        # },

        "use_cyclic_encoders": False,
    }
    return hyperparameters
