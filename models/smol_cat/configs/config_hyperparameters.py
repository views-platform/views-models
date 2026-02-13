def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.
    This configuration is "operational" so modifying these settings will impact the model's behavior during the training.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for training the model.
    """
    
    hyperparameters = {
        # ==============================================================================
        # CORE SETTINGS
        # ==============================================================================
        "steps": [*range(1, 36 + 1, 1)],
        "num_samples": 1,
        "mc_dropout": True,
        "random_state": 67,
        "n_jobs": 2,
        "n_epochs": 200,
        "optimizer_cls": "Adam",
        "output_chunk_length": 36,
        "output_chunk_shift": 0,
        "force_reset": True,
        
        # ==============================================================================
        # ARCHITECTURE (Best Run Configuration)
        # ==============================================================================
        "batch_size": 1024,
        "input_chunk_length": 72,           # Long context (6 years)
        "hidden_size": 256,                 # Wide layers
        "num_encoder_layers": 3,
        "num_decoder_layers": 2,
        "decoder_output_dim": 64,
        "temporal_width_past": 6,
        "temporal_width_future": 6,
        "temporal_hidden_size_past": 64,
        "temporal_hidden_size_future": 128,
        "temporal_decoder_hidden": 128,
        
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        "dropout": 0.05,                    # Low dropout to preserve rare signals
        "use_layer_norm": True,
        "use_reversible_instance_norm": True,
        "use_static_covariates": False,
        "weight_decay": 0,                  # No weight decay
        
        # ==============================================================================
        # OPTIMIZATION (CosineAnnealingWarmRestarts Strategy)
        # ==============================================================================
        "lr": 0.0009461214582864406,        # Swept learning rate
        "lr_scheduler_cls": "CosineAnnealingWarmRestarts",
        "lr_scheduler_T_0": 20,             # Initial restart cycle
        "lr_scheduler_T_mult": 2,           # Double cycle length after restart
        "lr_scheduler_eta_min": 1e-6,       # Minimum LR floor
        "gradient_clip_val": 1.5,
        
        # Early stopping
        "early_stopping_patience": 35,      # High patience to survive LR restarts
        "early_stopping_min_delta": 0.0001,
        
        # ==============================================================================
        # LOSS FUNCTION: WeightedPenaltyHuberLoss (Swept Weights)
        # ==============================================================================
        "loss_function": "WeightedPenaltyHuberLoss",
        
        # Huber delta (close to 1.0 = MSE-like behavior for most scaled data)
        "delta": 0.9658338566551232,
        
        # Zero threshold (scaled space ~0.037 corresponds to low fatality counts)
        "zero_threshold": 0.036517407890917314,
        
        # ADDITIVE WEIGHTS:
        # TN = 1.0 (base)
        # TP = 1.0 + 10.0 = 11.0
        # FP = 0.72 (Absolute - cheap to explore)
        # FN = 1.0 + 10.0 + 6.35 = 17.35
        "non_zero_weight": 10.0,
        "false_positive_weight": 0.7220071760903927,
        "false_negative_weight": 6.34931734289296,
        
        # ==============================================================================
        # SCALING MAPS
        # ==============================================================================
        "target_scaler": "AsinhTransform->MinMaxScaler",
        "feature_scaler": None,
        "feature_scaler_map": {
            "MinMaxScaler": [
                "lr_wdi_sl_tlf_totl_fe_zs",
                "lr_wdi_se_enr_prim_fm_zs",
                "lr_wdi_sp_urb_totl_in_zs",
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
            ],
            "RobustScaler->MinMaxScaler": [
                "lr_splag_1_ged_sb",
                "lr_splag_1_ged_ns",
                "lr_splag_1_ged_os",
            ],
            "AsinhTransform->MinMaxScaler": [
                "lr_ged_sb",
                "lr_ged_ns",
                "lr_ged_os",
                "lr_acled_sb",
                "lr_acled_os",
                "lr_wdi_sm_pop_refg_or",
                "lr_wdi_ny_gdp_mktp_kd",
                "lr_wdi_nv_agr_totl_kn",
            ],
            "StandardScaler->MinMaxScaler": [
                "lr_wdi_sm_pop_netm",
                "lr_wdi_dt_oda_odat_pc_zs",
                "lr_wdi_sp_pop_grow",
                "lr_wdi_ms_mil_xpnd_gd_zs",
                "lr_wdi_sp_dyn_imrt_fe_in",
                "lr_wdi_sh_sta_stnt_zs",
                "lr_wdi_sh_sta_maln_zs",
            ],
        },
    }

    return hyperparameters