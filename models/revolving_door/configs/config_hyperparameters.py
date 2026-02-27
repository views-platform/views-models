def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.
    This configuration is "operational" so modifying these settings will impact the model's behavior during the training.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for training the model, which determine the model's behavior during the training phase.
    """
    
    hyperparameters = {
        # Temporal
        "steps": [*range(1, 36 + 1, 1)],
        "input_chunk_length": 36,
        "output_chunk_length": 36,
        "output_chunk_shift": 0,
        "random_state": 67,
        
        # Inference
        "num_samples": 1,
        "mc_dropout": True,
        "n_jobs": -1,
        
        # Training
        "batch_size": 64,
        "n_epochs": 200,
        "early_stopping_patience": 30,
        "early_stopping_min_delta": 0.0001,
        "force_reset": True,
        
        # Optimizer
        "optimizer_cls": "Adam",
        "lr": 0.00016112813049684902,
        "weight_decay": 0.000001,
        "gradient_clip_val": 1.5,
        
        # LR Scheduler
        "lr_scheduler_cls": "CosineAnnealingWarmRestarts",
        "lr_scheduler_T_0": 25,
        "lr_scheduler_T_mult": 1,
        "lr_scheduler_eta_min": 0.000001,
        "lr_scheduler_kwargs": {
            "T_0": 25,
            "T_mult": 1,
            "eta_min": 0.000001,
        },
        "optimizer_kwargs": {
            "lr": 0.00016112813049684902,
            "weight_decay": 0.000001,
        },
        
        # Loss: MagnitudeAwareQuantileLoss
        "loss_function": "MagnitudeAwareQuantileLoss",
        "tau": 0.45214108890315474,
        "zero_threshold": 2.09,
        "non_zero_weight": 35.20259257390748,
        
        # Scaling
        "feature_scaler": None,
        "target_scaler": "AsinhTransform",
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
            "AsinhTransform": [
                "lr_acled_sb",
                "lr_acled_os",
                "lr_wdi_sm_pop_refg_or",
                "lr_wdi_ny_gdp_mktp_kd",
                "lr_wdi_nv_agr_totl_kn",
                "lr_splag_1_ged_sb",
                "lr_splag_1_ged_ns",
                "lr_splag_1_ged_os",
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
        
        # N-HiTS Architecture
        "num_stacks": 2,
        "num_blocks": 1,
        "num_layers": 2,
        "layer_widths": 512,
        "pooling_kernel_sizes": [[8], [1]],
        "n_freq_downsample": [[4], [1]],
        "max_pool_1d": True,
        "activation": "ReLU",
        "dropout": 0.15,
        "use_reversible_instance_norm": False,
    }

    return hyperparameters