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
        "mc_dropout": True,
        "random_state": 67,
        "batch_size": 128,
        "decoder_output_dim": 128,
        "dropout": 0.25,
        "early_stopping_min_delta": 0.0001,
        "early_stopping_patience": 30,
        "feature_scaler": None,
        "feature_scaler_map": {
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
            "MinMaxScaler": [
                "month",
                "lr_wdi_sl_tlf_totl_fe_zs",
                "lr_wdi_se_enr_prim_fm_zs",
                "lr_wdi_sp_urb_totl_in_zs",
                # V-Dem (all bounded 0-1)
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
                # Topics (probabilities 0-1)
                "lr_topic_ste_theta0",
                "lr_topic_ste_theta1",
                "lr_topic_ste_theta2",
                "lr_topic_ste_theta3",
                "lr_topic_ste_theta4",
                "lr_topic_ste_theta5",
                "lr_topic_ste_theta6",
            ],
        },
        "force_reset": True,
        "gradient_clip_val": 1,
        "hidden_size": 256,
        "input_chunk_length": 36,
        "loss_function": "NegativeBinomialLoss",
        "lr": 0.00010128431037201398,
        "lr_scheduler_cls": "CosineAnnealingWarmRestarts",
        "lr_scheduler_T_0": 25,
        "lr_scheduler_T_mult": 1,
        "lr_scheduler_eta_min": 1e-6,
        "n_epochs": 1,
        "n_jobs": -1,
        "num_decoder_layers": 2,
        "num_encoder_layers": 2,
        "optimizer_cls": "Adam",
        "output_chunk_shift": 0,
        "output_chunk_length": 36,
        "target_scaler": None,
        "temporal_decoder_hidden": 256,
        "temporal_hidden_size_future": 256,
        "temporal_hidden_size_past": 256,
        "temporal_width_future": 64,
        "temporal_width_past": 64,
        # ==============================================================================
        # TEMPORAL ENCODINGS (Position-based)
        # ==============================================================================
        # use_datetime_index: Convert views month_id to DatetimeIndex
        #   - Required for cyclic encoders (month, week, dayofweek sin/cos)
        #   - NOT required for position encoder (works with integer indices)
        # temporal_precision: Which views index type (month, week, day) - for DatetimeIndex conversion
        #
        # position encoder: relative position in sequence (0.0 to 1.0)
        #   - Works with ANY index type (int or datetime)
        #   - Provides temporal context without requiring DatetimeIndex
        #
        # Note: Cyclic encoding (sin/cos for month) requires DatetimeIndex which has
        # compatibility issues with Darts slicing. Seasonality is captured via raw
        # 'month' feature (MinMax scaled) in the queryset instead.
        "use_datetime_index": False,
        "temporal_precision": "month",  # month | week | day (for future use)
        "add_encoders": {
            "position": {"past": ["relative"], "future": ["relative"]},
        },
        "use_layer_norm": True,
        "use_reversible_instance_norm": False,
        "use_static_covariates": True,
        "weight_decay": 1e-6,
        # NegativeBinomialLoss parameters
        "alpha": 0.5249628138403757,
        "zero_threshold": 4,
        "false_negative_weight": 1,
        "false_positive_weight": 5.658671930633947,
        "learn_alpha": False,
        "inverse_transform": False,
    }

    return hyperparameters