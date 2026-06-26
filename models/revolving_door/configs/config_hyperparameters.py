def get_hp_config():
    hyperparameters = {
        # Temporal
        "steps": [*range(1, 36 + 1)],
        "input_chunk_length": 36,
        "output_chunk_length": 36,
        "output_chunk_shift": 0,
        "random_state": 67,
        "time_steps": 36,

        # Inference
        "num_samples": 1,
        "mc_dropout": False,
        "n_jobs": -1,

        # Training
        "batch_size": 128,                   
        "n_epochs": 300,
        "early_stopping_patience": 40,
        "early_stopping_min_delta": 0.001,
        "force_reset": True,

        # Optimizer
        "optimizer_cls": "AdamW",
        "lr": 0.0003,                          # Stable global learning rate for LeakyReLU at batch_size=128
        "weight_decay": 0.001,                  # Moderate decay to regularize weights
        "gradient_clip_val": 1.0,

        # LR Scheduler
        "lr_scheduler_cls": "ReduceLROnPlateau",
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 15,           # Allows multiple LR decays before early stopping (patience=40)
        "lr_scheduler_min_lr": 1e-6,
        "lr_scheduler_kwargs": {
            "mode": "min",
            "factor": 0.5,
            "patience": 15,                    # Consistent with global scheduler patience
            "min_lr": 1e-6,
            "cooldown": 5,                     # Give parameters time to settle post-decay
            "threshold": 0.005,                # Require clear improvement
            "threshold_mode": "rel",
            "monitor": "val_loss",
        },
        "optimizer_kwargs": {
            "lr": 0.0003,
            "weight_decay": 0.001,
        },

        # Loss & Scaling
        "loss_function": "SpotlightLossLogcosh",
        "non_zero_threshold": 0.88,
        "feature_scaler": None,
        "target_scaler": "AsinhTransform",
        "feature_scaler_map": {
            "AsinhTransform->MaxAbsScaler": [
                "lr_ged_ns", "lr_ged_os", "lr_ged_sb_delta", "lr_ged_ns_delta", "lr_ged_os_delta",
                "lr_acled_sb", "lr_acled_sb_count", "lr_acled_os",
                "lr_splag_1_ged_sb", "lr_splag_1_ged_ns", "lr_splag_1_ged_os",
                "lr_decay_ged_sb_5", "lr_decay_ged_sb_100", "lr_decay_ged_sb_500",
                "lr_decay_ged_os_5", "lr_decay_ged_os_100", "lr_decay_ged_ns_5", "lr_decay_ged_ns_100",
                "lr_decay_acled_sb_5", "lr_decay_acled_os_5", "lr_decay_acled_ns_5",
                "lr_splag_1_decay_ged_sb_5", "lr_splag_1_decay_ged_os_5", "lr_splag_1_decay_ged_ns_5",
                "lr_ged_sb_tlag_1", "lr_ged_sb_tlag_2", "lr_ged_sb_tlag_3", "lr_ged_sb_tlag_4", "lr_ged_sb_tlag_5", "lr_ged_sb_tlag_6", "lr_ged_os_tlag_1",
                "lr_topic_tokens_t1", "lr_topic_tokens_t2",
                "lr_topic_ste_theta4_stock_t1", "lr_topic_ste_theta4_stock_t2", "lr_topic_ste_theta4_stock_t13",
                "lr_topic_ste_theta2_stock_t1", "lr_topic_ste_theta2_stock_t2", "lr_topic_ste_theta2_stock_t13",
                "lr_topic_ste_theta4_stock_t1_splag", "lr_topic_ste_theta2_stock_t1_splag",
                "lr_wdi_sm_pop_refg_or", "lr_wdi_sm_pop_netm", "lr_wdi_dt_oda_odat_pc_zs", "lr_wdi_ms_mil_xpnd_gd_zs",
                "lr_wdi_sp_pop_grow", "lr_wdi_sp_urb_totl_in_zs", "lr_wdi_sp_dyn_imrt_fe_in", "lr_wdi_sh_sta_maln_zs",
                "lr_vdem_v2x_horacc", "lr_vdem_v2x_veracc", "lr_vdem_v2xnp_client", "lr_vdem_v2xnp_regcorr",
                "lr_vdem_v2xpe_exlgeo", "lr_vdem_v2xpe_exlsocgr", "lr_vdem_v2x_ex_party", "lr_vdem_v2x_ex_military",
                "lr_vdem_v2xeg_eqdr", "lr_vdem_v2xcl_prpty", "lr_vdem_v2xcl_dmove", "lr_vdem_v2x_clphy",
            ],
        },

        # N-HiTS Architecture
        "num_stacks": 3,
        "num_blocks": 1,
        "num_layers": 3,                       # Capacity depth (3 layers per block)
        "layer_widths": 256,                   # Increased capacity: uniform width of 256 across all blocks to capture unique country trajectories without collapsing to templates (original run notes validate uniform 256)
        "pooling_kernel_sizes": [[6], [3], [1]], # Coarse (6:1), medium (3:1), fine (1:1) pooling downsampling
        "n_freq_downsample": [[6], [2], [1]],    # Hierarchical frequency representation matching basis interpolation
        "activation": "GELU",                  
        "dropout": 0.25,                       # Standard regularization to prevent memorize-overfitting while preserving expressive capacity
        "use_static_covariates": True,
        "use_reversible_instance_norm": True,
        "max_pool_1d": False,                 
    }

    return hyperparameters