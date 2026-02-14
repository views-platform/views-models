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
        "n_jobs": -1,
        "n_epochs": 200,
        "optimizer_cls": "Adam",
        "output_chunk_length": 36,
        "output_chunk_shift": 0,
        "force_reset": True,
        
        # ==============================================================================
        # ARCHITECTURE (Best Run Configuration)
        # ==============================================================================
        "batch_size": 64,
        "input_chunk_length": 48,
        "hidden_size": 256,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "decoder_output_dim": 128,
        "temporal_width_past": 12,
        "temporal_width_future": 12,
        "temporal_hidden_size_past": 128,
        "temporal_hidden_size_future": 128,
        "temporal_decoder_hidden": 256,
        
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        "dropout": 0.05,
        "use_layer_norm": True,
        "use_reversible_instance_norm": False,
        "use_static_covariates": False,
        "weight_decay": 1e-6,
        
        # ==============================================================================
        # OPTIMIZATION (CosineAnnealingWarmRestarts Strategy)
        # ==============================================================================
        "lr": 0.001358054181340207,
        "lr_scheduler_cls": "CosineAnnealingWarmRestarts",
        "lr_scheduler_T_0": 50,
        "lr_scheduler_T_mult": 1,
        "lr_scheduler_eta_min": 1e-6,
        "gradient_clip_val": 1.5,
        
        # Early stopping
        "early_stopping_patience": 20,
        "early_stopping_min_delta": 0.0001,
        
        # ==============================================================================
        # LOSS FUNCTION: WeightedPenaltyHuberLoss (Swept Weights)
        # ==============================================================================
        "loss_function": "WeightedPenaltyHuberLoss",
        
        # Huber delta (asinh scale: ~1.2 means L2 for small conflicts, L1 for large)
        "delta": 1.1961636786267764,
        
        # Zero threshold (asinh scale: ~1.16 ≈ asinh(1.4) fatalities)
        "zero_threshold": 1.159666712548936,
        
        # ADDITIVE WEIGHTS:
        # TN = 1.0 (base)
        # TP = 1.0 + 25.0 = 26.0
        # FP = 0.456 (Absolute - cheap to explore)
        # FN = 1.0 + 25.0 + 16.3 = 42.3
        "non_zero_weight": 25,
        "false_positive_weight": 0.456053758405519,
        "false_negative_weight": 16.30133196931405,
        
        # ==============================================================================
        # SCALING MAPS
        # ==============================================================================
        "target_scaler": "AsinhTransform",
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
            "RobustScaler": [
                "lr_splag_1_ged_sb",
                "lr_splag_1_ged_ns",
                "lr_splag_1_ged_os",
            ],
            "AsinhTransform": [
                "lr_acled_sb",
                "lr_acled_os",
                "lr_wdi_sm_pop_refg_or",
                "lr_wdi_ny_gdp_mktp_kd",
                "lr_wdi_nv_agr_totl_kn",
            ],
            "StandardScaler": [
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