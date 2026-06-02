
def get_hp_config():
    """
    TCN hyperparameters for conflict forecasting.

    Architecture: Dilated causal TCN with residual connections + weight norm.
    Receptive field = 1 + 2*(kernel_size-1)*(dilation_base^num_layers - 1)/(dilation_base - 1)
                    = 1 + 2*2*(16-1)/1 = 61 timesteps (covers full input_chunk_length=48)

    Key constraint: output_chunk_length MUST be < input_chunk_length (Darts hard check).
    TCN is PastCovariatesTorchModel only — no future covariates, no static covariates.
    """

    hyperparameters = {
        # Temporal
        # input_chunk_length=48: 4 years of history. Must be > output_chunk_length.
        # output_chunk_length=36: produces 36-step forecast (matches steps).
        "steps": [*range(1, 36 + 1)],
        "input_chunk_length": 48,
        "output_chunk_length": 36,
        "output_chunk_shift": 0,
        "random_state": 67,
        "time_steps": 36,  # Checksum: Must match len(steps)

        # Inference
        "num_samples": 1,
        "mc_dropout": False,
        "n_jobs": -1,

        # Training
        "batch_size": 128,
        "n_epochs": 300,
        "early_stopping_patience": 20,
        "early_stopping_min_delta": 0.001,
        "force_reset": True,

        # Optimizer
        "optimizer_cls": "AdamW",
        "lr": 2e-4,
        "weight_decay": 1e-3,  # Increased to bound conv weights actively and prevent ReLu/residual explosions
        "gradient_clip_val": 1.0,

        # LR Scheduler
        "lr_scheduler_cls": "ReduceLROnPlateau",
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 7,
        "lr_scheduler_min_lr": 1e-6,
        "lr_scheduler_kwargs": {
            "mode": "min",
            "factor": 0.5,
            "patience": 7,
            "min_lr": 1e-6,
            "cooldown": 2,
            "threshold": 0.005,
            "threshold_mode": "rel",
        },
        "optimizer_kwargs": {
            "lr": 2e-4,
            "weight_decay": 1e-3,
        },

        # SpotlightLossLogcosh
        "loss_function": "SpotlightLossLogcosh",
        "non_zero_threshold": 0.88,

        # Scaling
        "feature_scaler": None,
        "target_scaler": "AsinhTransform",
        "feature_scaler_map": {
            "AsinhTransform->MinMaxScaler": [
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

                    # ln_ged temporal lags — explicit trajectory
                    "ln_ged_sb_tlag_1", "ln_ged_sb_tlag_2", "ln_ged_sb_tlag_3",
                    "ln_ged_sb_tlag_4", "ln_ged_sb_tlag_5", "ln_ged_sb_tlag_6",
                    "ln_ged_os_tlag_1",

                    # Topic/NLP features — monthly leading indicators
                    "lr_topic_tokens_t1", "lr_topic_tokens_t2",
                    "lr_topic_ste_theta4_stock_t1", "lr_topic_ste_theta4_stock_t2", "lr_topic_ste_theta4_stock_t13",
                    "lr_topic_ste_theta2_stock_t1", "lr_topic_ste_theta2_stock_t2", "lr_topic_ste_theta2_stock_t13",
                    "lr_topic_ste_theta4_stock_t1_splag", "lr_topic_ste_theta2_stock_t1_splag",

                    # WDI (8)
                    "lr_wdi_sm_pop_refg_or", "lr_wdi_sm_pop_netm",
                    "lr_wdi_dt_oda_odat_pc_zs", "lr_wdi_ms_mil_xpnd_gd_zs",
                    "lr_wdi_sp_pop_grow",
                    "lr_wdi_sp_urb_totl_in_zs",
                    "lr_wdi_sp_dyn_imrt_fe_in",
                    "lr_wdi_sh_sta_maln_zs",

                    # V-Dem (12)
                    "lr_vdem_v2x_horacc", "lr_vdem_v2x_veracc",
                    "lr_vdem_v2xnp_client", "lr_vdem_v2xnp_regcorr",
                    "lr_vdem_v2xpe_exlgeo", "lr_vdem_v2xpe_exlsocgr",
                    "lr_vdem_v2x_ex_party", "lr_vdem_v2x_ex_military",
                    "lr_vdem_v2xeg_eqdr",
                    "lr_vdem_v2xcl_prpty", "lr_vdem_v2xcl_dmove", "lr_vdem_v2x_clphy",
                ],
        },

        # TCN Architecture
        # 4 residual blocks with exponential dilation: d=[1,2,4,8]
        # RF = 1 + 2*(3-1)*(2^4 - 1)/(2-1) = 61 > input_chunk_length=48 ✓
        # Darts' TCN implementation lacks internal LayerNorm/BatchNorm inside the residual 
        # path, meaning ReLU outputs accumulate exponentially. To stop runaway predictions, we drop 
        # num_filters to 64 (limiting parallel paths), increase weight_decay considerably (1e-3) to bound 
        # conv filters, and drop dropout to 0.05 (too much dropout scales activations by 1/(1-p) during 
        # training which massively inflates residual accumulation).
        "kernel_size": 3,
        "num_filters": 64,
        "num_layers": 4,
        "dilation_base": 2,
        "weight_norm": True,
        "dropout": 0.05,
        "use_static_covariates": True,
        "use_reversible_instance_norm": True,
        "checkpoint_mode": "best",

        # Cyclic encoders inject sin/cos(month) as past covariates
        "use_cyclic_encoders": True,
    }

    return hyperparameters