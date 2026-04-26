
def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.
    This configuration is "operational" so modifying these settings will impact the model's behavior during the training.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for training the model, which determine the model's behavior during the training phase.
    """
    
    hyperparameters = {
        "steps": [*range(1, 36 + 1, 1)],
        "num_samples": 1,
        "mc_dropout": False,
        "detect_anomaly": False,
        "time_steps": 36,  # Checksum: Must match len(steps)
        "rolling_origin_stride": 1,

        "activation": "SwiGLU",
        "batch_size": 256,
        "d_model": 256,
        "dim_feedforward": 1024,
        "dropout": 0.15,
        "early_stopping_min_delta": 0.0001,
        "early_stopping_patience": 30,
        "feature_scaler": None,
        "feature_scaler_map": {
            "LogTransform": [
                "lr_wdi_sp_dyn_imrt_fe_in",
                "lr_wdi_sm_pop_refg_or",
                "lr_wdi_ny_gdp_mktp_kd",
                "lr_wdi_nv_agr_totl_kn",
                "lr_splag_1_ged_sb",
                "lr_splag_1_ged_ns",
                "lr_splag_1_ged_os",
                "lr_ged_ns",
                "lr_ged_os",
            ],
            "MinMaxScaler": [
                "lr_wdi_sh_sta_stnt_zs",
                "lr_wdi_sh_sta_maln_zs",
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
            "StandardScaler": [
                "lr_ged_sb_delta",
                "lr_ged_ns_delta",
                "lr_ged_os_delta",
                "lr_wdi_sm_pop_netm",
                "lr_wdi_dt_oda_odat_pc_zs",
                "lr_wdi_sp_pop_grow",
                "lr_wdi_ms_mil_xpnd_gd_zs",
            ],
        },
        "force_reset": True,
        "gradient_clip_val": 1,
        "input_chunk_length": 36,
        "loss_function": "PrismLoss",
        "delta": 0.153718655222389,
        "event_weight": 0.5,
        "dual_mean": False,
        "non_zero_threshold": 0.693,
        "lr": 0.0005635679955633913,
        "lr_scheduler_cls": "ReduceLROnPlateau",
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 15,
        "lr_scheduler_min_lr": 0.000001,
        "lr_scheduler_kwargs": {
            "mode": "min",
            "factor": 0.5,
            "min_lr": 0.000001,
            "monitor": "train_loss",
            "patience": 15,
        },
        "n_epochs": 300,
        "nhead": 4,
        "norm_type": "LayerNorm",
        "num_decoder_layers": 2,
        "num_encoder_layers": 2,
        "optimizer_cls": "AdamW",
        "optimizer_kwargs": {
            "lr": 0.0005635679955633913,
            "weight_decay": 0.0001,
        },
        "output_chunk_length": 36,
        "output_chunk_shift": 0,
        "random_state": 67,
        "target_scaler": "LogTransform",
        "use_reversible_instance_norm": True,
        "weight_decay": 0.0001,

        # Encoders
        "use_cyclic_encoders": True,

        # Prediction output format
        "prediction_format": "dataframe",
    }


    return hyperparameters