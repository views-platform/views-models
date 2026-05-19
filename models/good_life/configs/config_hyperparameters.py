
def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.
    This configuration is "operational" so modifying these settings will impact the model's behavior during the training.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for training the model, which determine the model's behavior during the training phase.
    """
    
    hyperparameters = {
        "steps": [*range(1, 36 + 1)],
        "num_samples": 1,
        "mc_dropout": False,
        "detect_anomaly": False,
        "time_steps": 36,  # Checksum: Must match len(steps)

        "activation": "GELU",
        "batch_size": 128,
        "d_model": 128,
        "dim_feedforward": 512,
        "dropout": 0.25,
        "early_stopping_min_delta": 0.001,
        "early_stopping_patience": 35,
        "feature_scaler": None,
        "feature_scaler_map": {
            "AsinhTransform->MaxAbsScaler": [
                "lr_splag_1_ged_sb", "lr_splag_1_ged_ns", "lr_splag_1_ged_os",
                "lr_ged_ns", "lr_ged_os",
                "lr_ged_sb_delta", "lr_ged_ns_delta", "lr_ged_os_delta",
                "lr_acled_sb", "lr_acled_sb_count", "lr_acled_os",
                "lr_wdi_ny_gdp_mktp_kd", "lr_wdi_nv_agr_totl_kn",
                "lr_wdi_sm_pop_refg_or", "lr_wdi_sm_pop_netm",
                "lr_wdi_dt_oda_odat_pc_zs",
                "lr_wdi_ms_mil_xpnd_gd_zs",
                "lr_vdem_v2x_horacc", "lr_vdem_v2x_veracc", "lr_vdem_v2x_diagacc",
                "lr_vdem_v2xnp_client", "lr_vdem_v2xnp_regcorr",
                "lr_vdem_v2xpe_exlpol", "lr_vdem_v2xpe_exlgeo",
                "lr_vdem_v2xpe_exlgender", "lr_vdem_v2xpe_exlsocgr",
                "lr_vdem_v2x_divparctrl", "lr_vdem_v2x_ex_party",
                "lr_vdem_v2x_ex_military", "lr_vdem_v2x_genpp",
                "lr_vdem_v2xeg_eqdr", "lr_vdem_v2xcl_prpty",
                "lr_vdem_v2xeg_eqprotec", "lr_vdem_v2xcl_dmove",
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
        "force_reset": True,
        "gradient_clip_val": 2,
        "input_chunk_length": 36,
        "loss_function": "SpotlightLossLogcosh",
        "delta": 0.12403671041645452,
        "non_zero_threshold": 0.88,
        "lr": 0.0003,
        "lr_scheduler_cls": "ReduceLROnPlateau",
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 12,
        "lr_scheduler_min_lr": 0.000001,
        "lr_scheduler_kwargs": {
            "mode": "min",
            "factor": 0.5,
            "patience": 12,
            "min_lr": 0.000001,
            "threshold": 0.01,
            "threshold_mode": "rel",
            "cooldown": 3,
        },
        "n_epochs": 300,
        "nhead": 4,
        "norm_type": "LayerNorm",
        "num_decoder_layers": 2,
        "num_encoder_layers": 3,
        "optimizer_cls": "AdamW",
        "optimizer_kwargs": {
            "lr": 0.0003,
            "weight_decay": 0.00002,
        },
        "output_chunk_length": 36,
        "output_chunk_shift": 0,
        "random_state": 67,
        "target_scaler": "AsinhTransform",
        "use_reversible_instance_norm": True,
        "weight_decay": 0.00002,

        # Encoders
        "use_cyclic_encoders": True,
        "use_static_covariates": True,
        # "static_covariate_stats": {"transform": "AsinhTransform"},
    }


    return hyperparameters