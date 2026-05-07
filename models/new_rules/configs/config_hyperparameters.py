def get_hp_config():
    """
    https://wandb.ai/views_pipeline/new_rules_nbeats_shadow_20260505_C_sweep/runs/gdltp3pj
    """ 

    hyperparameters = {
        # --- Forecast horizon ---
        "steps": [*range(1, 36 + 1)],

        # --- Architecture ---
        "generic_architecture": True,
        "num_stacks": 2,
        "num_blocks": 3,
        "num_layers": 3,
        "layer_widths": 128,
        "expansion_coefficient_dim": 128,
        "trend_polynomial_degree": 2,
        "activation": "GELU",
        "dropout": 0.15,
        "batch_norm": False,
        "use_reversible_instance_norm": True,
        "use_static_covariates": True,
        "use_cyclic_encoders": True,

        # --- Input / output structure ---
        "input_chunk_length": 36,
        "output_chunk_length": 36,
        "output_chunk_shift": 0,

        # --- Training ---
        "batch_size": 128,
        "n_epochs": 300,
        "early_stopping_patience": 35,
        "early_stopping_min_delta": 0.001,
        "force_reset": True,

        # --- Optimizer ---
        "optimizer_cls": "AdamW",
        "lr": 0.0005,
        "weight_decay": 0.0002,
        "gradient_clip_val": 3,
        "optimizer_kwargs": {
            "lr": 0.0005,
            "weight_decay": 0.0002,
        },

        # --- LR Scheduler ---
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

        # --- Scaling ---
        "target_scaler": "AsinhTransform",
        "feature_scaler": None,
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

        # --- Loss: SpotlightLoss v36 ---
        "loss_function": "SpotlightLossLogcosh",
        "delta": 0.01534983792365936,
        "non_zero_threshold": 0.88,  # asinh(1) ≈ 0.88 in asinh space (1 battle death)

        # --- Prediction ---
        "likelihood": None,
        "num_samples": 1,
        "mc_dropout": False,

        # --- Other ---
        "random_state": 67,
        "time_steps": 36,  # Checksum: Must match len(steps)
        "rolling_origin_stride": 1,
        "prediction_format": "dataframe",
        "static_covariate_stats": {"transform": "AsinhTransform->MaxAbsScaler"},

        # --- other ---
        "n_jobs": -1
    }

    return hyperparameters

