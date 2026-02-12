def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.
    This configuration is "operational" so modifying these settings will impact the model's behavior during the training.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for training the model, which determine the model's behavior during the training phase.
    """
    
    hyperparameters = {
        # Sweep best run hyperparameters (smol_cat_tide_v3_mtd)
        "steps": [*range(1, 36 + 1, 1)],
        "num_samples": 1,
        "mc_dropout": True,
        "random_state": 67,
        "n_jobs": -1,
        "n_epochs": 200,
        "optimizer_cls": "Adam",
        "output_chunk_length": 36,
        "output_chunk_shift": 0,
        "force_reset": True,
        
        # Architecture
        "batch_size": 1024,
        "input_chunk_length": 24,
        "hidden_size": 32,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "decoder_output_dim": 64,
        "temporal_width_past": 4,
        "temporal_width_future": 12,
        "temporal_hidden_size_past": 64,
        "temporal_hidden_size_future": 64,
        "temporal_decoder_hidden": 256,
        
        # Regularization
        "dropout": 0.15,
        "use_layer_norm": True,
        "use_reversible_instance_norm": True,
        "use_static_covariates": True,
        "weight_decay": 1e-5,
        
        # Learning rate schedule
        "lr": 0.00008292387485646309,
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 10,
        "lr_scheduler_min_lr": 1e-6,
        "gradient_clip_val": 1.5,
        
        # Early stopping
        "early_stopping_patience": 20,
        "early_stopping_min_delta": 0.0001,
        
        # Loss function: WeightedPenaltyHuberLoss
        "loss_function": "WeightedPenaltyHuberLoss",
        "delta": 0.9595721846790456,
        "zero_threshold": 0.0883088468236434,
        "non_zero_weight": 10.0,
        "false_positive_weight": 2.0621337209758197,
        "false_negative_weight": 36.56106010462274,
        
        # Scaling
        "target_scaler": "AsinhTransform->MinMaxScaler",
        "feature_scaler": None,
        "feature_scaler_map": {
            "MinMaxScaler": [
                "lr_wdi_sl_tlf_totl_fe_zs",
                "lr_wdi_se_enr_prim_fm_zs",
                "lr_wdi_sp_urb_totl_in_zs",
                "lr_wdi_sh_sta_maln_zs",
                "lr_wdi_sh_sta_stnt_zs",
                "lr_wdi_dt_oda_odat_pc_zs",
                "lr_wdi_ms_mil_xpnd_gd_zs",
                "lr_vdem_v2x_horacc",
                "lr_vdem_v2xnp_client",
                "lr_vdem_v2x_veracc",
                "lr_vdem_v2x_divparctrl",
                "lr_vdem_v2xpe_exlpol",
                "lr_vdem_v2x_diagacc",
                "lr_vdem_v2xpe_exlgeo",
                "lr_vdem_v2xpe_exlgender",
                "lr_vdem_v2xpe_exlsocgr",
                "lr_vdem_v2x_ex_party",
                "lr_vdem_v2x_genpp",
                "lr_vdem_v2xeg_eqdr",
                "lr_vdem_v2xcl_prpty",
                "lr_vdem_v2xeg_eqprotec",
                "lr_vdem_v2x_ex_military",
                "lr_vdem_v2xcl_dmove",
                "lr_vdem_v2x_clphy",
                "lr_vdem_v2x_hosabort",
                "lr_vdem_v2xnp_regcorr",
                "lr_topic_ste_theta0",
                "lr_topic_ste_theta1",
                "lr_topic_ste_theta2",
                "lr_topic_ste_theta3",
                "lr_topic_ste_theta4",
                "lr_topic_ste_theta5",
                "lr_topic_ste_theta6",
                "lr_topic_ste_theta7",
                "lr_topic_ste_theta8",
                "lr_topic_ste_theta9",
                "lr_topic_ste_theta10",
                "lr_topic_ste_theta11",
                "lr_topic_ste_theta12",
                "lr_topic_ste_theta13",
                "lr_topic_ste_theta14",
                "lr_topic_ste_theta0_stock_t1_splag",
                "lr_topic_ste_theta1_stock_t1_splag",
                "lr_topic_ste_theta2_stock_t1_splag",
                "lr_topic_ste_theta3_stock_t1_splag",
                "lr_topic_ste_theta4_stock_t1_splag",
                "lr_topic_ste_theta5_stock_t1_splag",
                "lr_topic_ste_theta6_stock_t1_splag",
                "lr_topic_ste_theta7_stock_t1_splag",
                "lr_topic_ste_theta8_stock_t1_splag",
                "lr_topic_ste_theta9_stock_t1_splag",
                "lr_topic_ste_theta10_stock_t1_splag",
                "lr_topic_ste_theta11_stock_t1_splag",
                "lr_topic_ste_theta12_stock_t1_splag",
                "lr_topic_ste_theta13_stock_t1_splag",
                "lr_topic_ste_theta14_stock_t1_splag",
                "lr_wdi_sp_pop_grow",
            ],
            "AsinhTransform->MinMaxScaler": [
                "lr_ged_sb",
                "lr_ged_ns",
                "lr_ged_os",
                "lr_acled_sb",
                "lr_acled_sb_count",
                "lr_acled_os",
                "lr_splag_1_ged_sb",
                "lr_splag_1_ged_os",
                "lr_splag_1_ged_ns",
                "lr_wdi_ny_gdp_mktp_kd",
                "lr_wdi_nv_agr_totl_kn",
                "lr_wdi_sm_pop_netm",
                "lr_wdi_sm_pop_refg_or",
                "lr_wdi_sp_dyn_imrt_fe_in",
                "lr_topic_tokens_t1",
                "lr_topic_tokens_t1_splag",
            ],
        },
    }

    return hyperparameters
