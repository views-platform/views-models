def get_hp_config():
    """
    N-HiTS hyperparameters from SpotlightLossLogcosh sweep best run.
    https://wandb.ai/views_pipeline/revolving_door_nhits_spotlight_v11_3_sweep/runs/p89rxmzk
    Returns:
    - hyperparameters (dict): Training configuration dictionary.
    """

    hyperparameters = {
        # Temporal
        "steps": [*range(1, 36 + 1)],
        "input_chunk_length": 36,
        "output_chunk_length": 36,
        "output_chunk_shift": 0,
        "random_state": 67,
        "time_steps": 36,  # Checksum: Must match len(steps)
        "rolling_origin_stride": 1,
        "prediction_format": "dataframe",

        # Inference
        "num_samples": 1,
        "mc_dropout": False,
        "n_jobs": -1,

        # Training
        "batch_size": 128,
        "n_epochs": 300,
        "early_stopping_patience": 35,
        "early_stopping_min_delta": 0.001,
        "force_reset": True,

        # Optimizer
        "optimizer_cls": "AdamW",
        "lr": 0.0005,
        "weight_decay": 0.00005,
        "gradient_clip_val": 3,

        # LR Scheduler
        "lr_scheduler_cls": "ReduceLROnPlateau",
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 12,
        "lr_scheduler_min_lr": 1e-6,
        "lr_scheduler_kwargs": {
            "mode": "min",
            "factor": 0.5,
            "patience": 12,
            "min_lr": 1e-6,
            "cooldown": 3,
            "threshold": 0.01,
            "threshold_mode": "rel",
        },
        "optimizer_kwargs": {
            "lr": 0.0005,
            "weight_decay": 0.00005,
        },

        # SpotlightLossLogcosh: logcosh base shape (gradient saturates at ±1)
        # Safe for basis-expansion architectures — bounded gradients prevent
        # learned interpolation coefficients from growing unbounded.
        "loss_function": "SpotlightLossLogcosh",
        "delta": 0.041685644972051974,
        "non_zero_threshold": 0.88,

        # Scaling
        "feature_scaler": None,
        "target_scaler": "AsinhTransform",
        "feature_scaler_map": {
            "AsinhTransform->MaxAbsScaler": [
                "lr_splag_1_ged_sb",
                "lr_splag_1_ged_ns",
                "lr_splag_1_ged_os",
                "lr_ged_ns",
                "lr_ged_os",
                "lr_ged_sb_delta",
                "lr_ged_ns_delta",
                "lr_ged_os_delta",
                "lr_acled_sb",
                "lr_acled_sb_count",
                "lr_acled_os",
                "lr_wdi_ny_gdp_mktp_kd",
                "lr_wdi_nv_agr_totl_kn",
                "lr_wdi_sm_pop_refg_or",
                "lr_wdi_sm_pop_netm",
                "lr_wdi_dt_oda_odat_pc_zs",
                "lr_wdi_ms_mil_xpnd_gd_zs",
                "lr_vdem_v2x_horacc",
                "lr_vdem_v2x_veracc",
                "lr_vdem_v2x_diagacc",
                "lr_vdem_v2xnp_client",
                "lr_vdem_v2xnp_regcorr",
                "lr_vdem_v2xpe_exlpol",
                "lr_vdem_v2xpe_exlgeo",
                "lr_vdem_v2xpe_exlgender",
                "lr_vdem_v2xpe_exlsocgr",
                "lr_vdem_v2x_divparctrl",
                "lr_vdem_v2x_ex_party",
                "lr_vdem_v2x_ex_military",
                "lr_vdem_v2x_genpp",
                "lr_vdem_v2xeg_eqdr",
                "lr_vdem_v2xcl_prpty",
                "lr_vdem_v2xeg_eqprotec",
                "lr_vdem_v2xcl_dmove",
                "lr_vdem_v2x_clphy",
                "lr_wdi_sp_pop_grow",
                "lr_wdi_sl_tlf_totl_fe_zs",
                "lr_wdi_se_enr_prim_fm_zs",
                "lr_wdi_sp_urb_totl_in_zs",
                "lr_wdi_sp_dyn_imrt_fe_in",
                "lr_wdi_sh_sta_stnt_zs",
                "lr_wdi_sh_sta_maln_zs",
            ],
        },

        # N-HiTS Architecture
        # Inverted pyramid: coarse stack is narrow (64), fine stack is wide (256).
        # This matches the decomposition task — the coarse stack only needs to capture
        # slow level shifts and physically cannot build complex crisis trend extrapolations
        # with only 64-wide FC layers. The fine stack gets 256 width to fit spike residuals
        # accurately, which drives MSLE down. Reverting to this from [160,80,64] which
        # gave the coarse stack too much capacity and caused Niger-type runaway.
        # Pooling: [4,2,1] → 9, 18, 36 FC inputs. Safe with 64-wide coarse stack +
        # RevIN sigma cap (sigma_raw now capped to 5× batch mean, so even if n_freq=4
        # extrapolates trend, the denorm multiplier is bounded).
        "num_stacks": 3,
        "num_blocks": 1,
        "num_layers": 4,
        "layer_widths": [64, 128, 256],
        "pooling_kernel_sizes": [[4], [2], [1]],
        "n_freq_downsample": [[4], [2], [1]],
        "max_pool_1d": True,
        "activation": "GELU",
        "dropout": 0.15,
        "use_static_covariates": True,
        "use_reversible_instance_norm": True,
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