
def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.
    This configuration is "operational" so modifying these settings will impact the model's behavior during the training.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for training the model, which determine the model's behavior during the training phase.
    """
    # r8
    hyperparameters = {
        "steps": [*range(1, 36 + 1)],
        "num_samples": 1,
        "mc_dropout": False,
        "detect_anomaly": False,
        "time_steps": 36,  # Checksum: Must match len(steps)

        "activation": "SwiGLU",
        "batch_size": 128,
        "d_model": 256,
        "dim_feedforward": 1024,
        "dropout": 0.15,
        "early_stopping_min_delta": 0.001,
        "early_stopping_patience": 20,
        "feature_scaler": None,
        "feature_scaler_map": {
            "AsinhTransform->MaxAbsScaler": [
                    # Conflict counts + deltas + spatial lags
                    "lr_ged_ns", "lr_ged_os",
                    "lr_ged_sb_delta", "lr_ged_ns_delta", "lr_ged_os_delta",
                    "lr_acled_sb", "lr_acled_sb_count", "lr_acled_os",
                    "lr_splag_1_ged_sb", "lr_splag_1_ged_ns", "lr_splag_1_ged_os",

                    # Decay features — conflict regime memory ∈ [0,1]
                    "lr_decay_ged_sb_5", "lr_decay_ged_sb_100", "lr_decay_ged_sb_500",
                    "lr_decay_ged_os_5", "lr_decay_ged_os_100",
                    "lr_decay_ged_ns_5", "lr_decay_ged_ns_100",
                    "lr_decay_acled_sb_5", "lr_decay_acled_os_5", "lr_decay_acled_ns_5",
                    "lr_splag_1_decay_ged_sb_5", "lr_splag_1_decay_ged_os_5", "lr_splag_1_decay_ged_ns_5",

                    # lr_ged temporal lags — explicit trajectory for TiDE (no recurrence)
                    "lr_ged_sb_tlag_1", "lr_ged_sb_tlag_2", "lr_ged_sb_tlag_3",
                    "lr_ged_sb_tlag_4", "lr_ged_sb_tlag_5", "lr_ged_sb_tlag_6",
                    "lr_ged_os_tlag_1",

                    # Topic/NLP features — monthly leading indicators
                    "lr_topic_tokens_t1", "lr_topic_tokens_t2",
                    "lr_topic_ste_theta4_stock_t1", "lr_topic_ste_theta4_stock_t2", "lr_topic_ste_theta4_stock_t13",
                    "lr_topic_ste_theta2_stock_t1", "lr_topic_ste_theta2_stock_t2", "lr_topic_ste_theta2_stock_t13",
                    "lr_topic_ste_theta4_stock_t1_splag", "lr_topic_ste_theta2_stock_t1_splag",

                    # WDI (8 with static covs)
                    "lr_wdi_sm_pop_refg_or", "lr_wdi_sm_pop_netm",
                    "lr_wdi_dt_oda_odat_pc_zs", "lr_wdi_ms_mil_xpnd_gd_zs",
                    "lr_wdi_sp_pop_grow",
                    "lr_wdi_sp_urb_totl_in_zs",
                    "lr_wdi_sp_dyn_imrt_fe_in",
                    "lr_wdi_sh_sta_maln_zs",

                    # V-Dem (12 — pruned of redundant accountability/exclusion)
                    "lr_vdem_v2x_horacc", "lr_vdem_v2x_veracc",
                    "lr_vdem_v2xnp_client", "lr_vdem_v2xnp_regcorr",
                    "lr_vdem_v2xpe_exlgeo", "lr_vdem_v2xpe_exlsocgr",
                    "lr_vdem_v2x_ex_party", "lr_vdem_v2x_ex_military",
                    "lr_vdem_v2xeg_eqdr",
                    "lr_vdem_v2xcl_prpty", "lr_vdem_v2xcl_dmove", "lr_vdem_v2x_clphy",
                ],
        },
        "force_reset": True,
        "gradient_clip_val": 50.0,
        "input_chunk_length": 36,
        "loss_function": "SpotlightLossLogcosh",
        "non_zero_threshold": 0.88,
        "lr": 5e-4,
        "lr_scheduler_cls": "ReduceLROnPlateau",
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 15,
        "lr_scheduler_min_lr": 1e-6,
        "lr_scheduler_kwargs": {
            "mode": "min",
            "factor": 0.5,
            "patience": 10,
            "min_lr": 1e-6,
            "cooldown": 2,
            "threshold": 0.01,
            "threshold_mode": "rel",
        },
        "n_epochs": 300,
        # Deep encoder/decoder stacks allow more refined temporal mixing.
        "nhead": 8,
        "norm_type": "LayerNorm",
        "num_decoder_layers": 5,
        "num_encoder_layers": 3,
        "optimizer_cls": "AdamW",
        "optimizer_kwargs": {
            "lr": 5e-4,
            "weight_decay": 1e-4,
        },
        "output_chunk_length": 36,
        "output_chunk_shift": 0,
        "random_state": 67,
        "target_scaler": "AsinhTransform",
        "use_reversible_instance_norm": True,
        "weight_decay": 1e-4,

        # Encoders
        "use_cyclic_encoders": True,
        "use_static_covariates": True,
        # "static_covariate_stats": {"transform": "AsinhTransform"},
    }


    return hyperparameters