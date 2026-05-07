
def get_hp_config():
    """
    TSMixer hyperparameters
    https://wandb.ai/views_pipeline/elastic_heart_tsmixer_shadow_20260505_A_sweep/runs/w10gy1p7
    """

    hyperparameters = {
        # Temporal
        "steps": [*range(1, 36 + 1, 1)],
        "input_chunk_length": 36,
        "output_chunk_length": 36,
        "output_chunk_shift": 0,
        "random_state": 67,
        "time_steps": 36,  # Checksum: Must match len(steps)
        "rolling_origin_stride": 1,

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
        "weight_decay": 1e-4,
        "gradient_clip_val": 1.5,

        # LR Scheduler
        "lr_scheduler_cls": "ReduceLROnPlateau",
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 8,
        "lr_scheduler_min_lr": 1e-6,
        "lr_scheduler_kwargs": {
            "mode": "min",
            "factor": 0.5,
            "patience": 8,
            "min_lr": 1e-6,
            "cooldown": 3,
            "threshold": 0.01,
            "threshold_mode": "rel",
        },
        "optimizer_kwargs": {
            "lr": 0.0005,
            "weight_decay": 1e-4,
        },
        "checkpoint_mode": "best",
        "loss_function": "SpotlightLossLogcosh",
        "delta": 0.05,
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

        # TSMixer Architecture
        # 2 blocks × 256 width: the channel-mixing MLP must route 47 features —
        # at h=128 pred_sanity/std collapses to 0.68 (vs 1.6 at h=256) because
        # the mixer can't differentiate feature interactions, squeezing all
        # predictions toward zero in normalized space. RevIN denorm then
        # amplifies these uniformly-positive values by sigma_raw (up to ~39
        # post-cap) → overpredict on every conflict country. h=256 gives
        # sufficient width for feature routing; overfitting is controlled by
        # weight_decay=1e-4 + dropout=0.25 + delta=0.05 spectral forcing.
        "num_blocks": 2,
        "hidden_size": 256,
        "ff_size": 256,
        "activation": "GELU",
        "norm_type": "LayerNorm",
        "normalize_before": True,
        "dropout": 0.25,
        "use_static_covariates": True,
        "use_reversible_instance_norm": True,

        "static_covariate_stats": {
            "transform": "AsinhTransform->MaxAbsScaler",
            "stats": ["sparsity", "trend"],
        },

        "use_cyclic_encoders": True,

        # Prediction output format
        "prediction_format": "dataframe",
    }
    return hyperparameters
