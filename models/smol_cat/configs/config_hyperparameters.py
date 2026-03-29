def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.
    This configuration is "operational" so modifying these settings will impact the model's behavior during the training.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for training the model, which determine the model's behavior during the training phase.
    """
    
    hyperparameters = {
        # Steps
        "steps": [*range(1, 36 + 1, 1)],
        "n_jobs": -1,

        # TiDE Architecture
        "input_chunk_length": 36,
        "output_chunk_length": 36,
        "output_chunk_shift": 0,
        "hidden_size": 256,
        "decoder_output_dim": 64,
        "temporal_decoder_hidden": 256,
        "temporal_width_past": 12,
        "temporal_width_future": 36,
        "temporal_hidden_size_past": 64,
        "temporal_hidden_size_future": 64,
        "num_encoder_layers": 1,
        "num_decoder_layers": 3,
        "use_layer_norm": True,
        "use_reversible_instance_norm": False,
        "dropout": 0.14553213915789748,
        "use_static_covariates": True,

        # Training
        "n_epochs": 300,
        "batch_size": 64,
        "random_state": 67,
        "force_reset": True,

        # Optimizer
        "optimizer_cls": "AdamW",
        "lr": 0.000030430804224905937,
        "weight_decay": 0.000005,

        # LR Scheduler
        "lr_scheduler_cls": "CosineAnnealingWarmRestarts",
        "lr_scheduler_T_0": 30,
        "lr_scheduler_T_mult": 2,
        "lr_scheduler_eta_min": 0.000001,

        # Trainer
        "gradient_clip_val": 3,
        "early_stopping_patience": 40,
        "early_stopping_min_delta": 0.0001,

        # Loss
        "loss_function": "SpotlightLoss",
        "alpha": 0.38723450371755985,
        "beta": 0.23550867726497013,
        "kappa": 12.486948159474515,
        "gamma": 0.1288880350112033,

        # Prediction
        "likelihood": None,
        "num_samples": 1,
        "mc_dropout": False,

        # Scalers
        "target_scaler": "AsinhTransform",
        "feature_scaler": None,
        "feature_scaler_map": {
            "MinMaxScaler": [
                "lr_wdi_sl_tlf_totl_fe_zs",
                "lr_wdi_se_enr_prim_fm_zs",
                "lr_wdi_sp_urb_totl_in_zs",
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
            ],
            "AsinhTransform": [
                "lr_wdi_sm_pop_refg_or",
                "lr_wdi_ny_gdp_mktp_kd",
                "lr_wdi_nv_agr_totl_kn",
                "lr_splag_1_ged_sb",
                "lr_splag_1_ged_ns",
                "lr_splag_1_ged_os",
            ],
            "StandardScaler": [
                "lr_ged_sb_delta",
                "lr_ged_ns_delta",
                "lr_ged_os_delta",
                "lr_wdi_sm_pop_netm",
                "lr_wdi_dt_oda_odat_pc_zs",
                "lr_wdi_sp_pop_grow",
                "lr_wdi_ms_mil_xpnd_gd_zs",
                "lr_wdi_sp_dyn_imrt_fe_in",
                "lr_wdi_sh_sta_stnt_zs",
                "lr_wdi_sh_sta_maln_zs",
            ],
        },

        # Encoders
        "use_cyclic_encoders": True,
    }

    return hyperparameters