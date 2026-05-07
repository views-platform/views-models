def get_hp_config():
    """
    TFT hyperparameters from SpotlightLossLogcosh sweep best run.
    """
    
    hyperparameters = {
        # Temporal
        "steps": [*range(1, 36 + 1, 1)],
        "time_steps": 36,  # Checksum: Must match len(steps)
        "rolling_origin_stride": 1,
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
        "n_epochs": 300,
        "early_stopping_patience": 15,
        "early_stopping_min_delta": 0.001,
        "force_reset": True,

        # Optimizer
        "optimizer_cls": "AdamW",
        "lr": 0.0005,
        "weight_decay": 0.001,
        "gradient_clip_val": 1.5,
        "optimizer_kwargs": {
            "lr": 0.0005,
            "weight_decay": 0.001,
        },

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
            "cooldown": 3,
            "threshold": 0.01,
            "threshold_mode": "rel",
        },

        # Loss
        "loss_function": "SpotlightLossLogcosh",
        "delta": 0.029357346926395782,
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

        # TFT Architecture
        "hidden_size": 128,
        "lstm_layers": 1,
        "num_attention_heads": 2,
        "full_attention": True,
        "feed_forward": "GatedResidualNetwork",
        "hidden_continuous_size": 32,
        "dropout": 0.35,
        "norm_type": "LayerNorm",
        "add_relative_index": True,
        "skip_interpolation": True,
        "categorical_embedding_sizes": {},
        "use_static_covariates": True,
        "use_reversible_instance_norm": True,
        "likelihood": None,

        "static_covariate_stats": {
            "transform": "AsinhTransform->MaxAbsScaler",
            "stats": ["sparsity", "trend"],
        },

        # Encoders
        "use_cyclic_encoders": True,

        # Prediction output format
        "prediction_format": "dataframe",
    }
    return hyperparameters
