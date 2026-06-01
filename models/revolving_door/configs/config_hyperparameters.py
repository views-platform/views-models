def get_hp_config():
    """
    N-HiTS hyperparameters from SpotlightLossLogcosh sweep best run.
    https://wandb.ai/views_pipeline/revolving_door_nhits_spotlight_v11_3_sweep/runs/p89rxmzk
    Returns:
    - hyperparameters (dict): Training configuration dictionary.
    """
    # r5
    hyperparameters = {
        # Temporal
        "steps": [*range(1, 36 + 1)],
        "input_chunk_length": 36,
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
        "early_stopping_patience": 35,
        "early_stopping_min_delta": 0.001,
        "force_reset": True,

        # Optimizer
        "optimizer_cls": "AdamW",
        "lr": 2e-4,
        "weight_decay": 1e-4,
        "gradient_clip_val": 50,

        # LR Scheduler
        "lr_scheduler_cls": "ReduceLROnPlateau",
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 12,
        "lr_scheduler_min_lr": 1e-6,
        "lr_scheduler_kwargs": {
            "mode": "min",
            "factor": 0.5,
            "patience": 12,
            "min_lr": 1e-6,
            "cooldown": 3,
            "threshold": 0.01,
            "threshold_mode": "rel",
        },
        "optimizer_kwargs": {
            "lr": 2e-4,
            "weight_decay": 1e-4,
        },

        # SpotlightLossLogcosh: logcosh base shape (gradient saturates at ±1)
        # Safe for basis-expansion architectures — bounded gradients prevent
        # learned interpolation coefficients from growing unbounded.
        "loss_function": "SpotlightLossLogcosh",
        "delta": 0.041685644972051974,
        "non_zero_threshold": 0.88,

        # Scaling
        "feature_scaler": None,
        "target_scaler": "AsinhTransform",
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

                    # ln_ged temporal lags — explicit trajectory for TiDE (no recurrence)
                    "ln_ged_sb_tlag_1", "ln_ged_sb_tlag_2", "ln_ged_sb_tlag_3",
                    "ln_ged_sb_tlag_4", "ln_ged_sb_tlag_5", "ln_ged_sb_tlag_6",
                    "ln_ged_os_tlag_1",

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

        # N-HiTS Architecture
        # Tanh activation bounds all hidden states to [-1,1], mechanically limiting
        # the forecast projection magnitude before RevIN denormalization.
        # Single block per stack (3 additive contributions total, not 6) reduces
        # cumulative output amplitude. Shallow blocks (2 layers) avoid vanishing
        # gradients from Tanh while keeping training stable.
        # Coarse stack: pool×6 + downsample×6 → 6 FC inputs, 6 forecast coefficients.
        # Very constrained: can only learn slow trends, not spike-scale extrapolation.
        # Fine stack: pool×1 + downsample×1 → 36 FC inputs, 36 forecast coefficients.
        # Full resolution for spike timing detail.
        # Widths increased: fine stack (2016→256) relieves the 10× compression
        # bottleneck that prevented event-scale representation.
        "num_stacks": 4,
        "num_blocks": 4,
        "num_layers": 3,
        "layer_widths": [512, 512, 512, 512],
        # "pooling_kernel_sizes": [[4, 4], [2, 2], [1, 1]],
        # "n_freq_downsample": [[4, 4], [2, 2], [1, 1]],
        "pooling_kernel_sizes": None,
        "n_freq_downsample": None,
        "max_pool_1d": False,
        "activation": "Tanh",
        "dropout": 0.25,
        "use_static_covariates": True,
        "use_reversible_instance_norm": True,
        "checkpoint_mode": "best",
        # "static_covariate_stats": {
        #     "transform": "AsinhTransform->MaxAbsScaler",
        #     "inject": False,
        # },
        # Temporal Encodings
        # ModelCatalog reads this flag and injects the appropriate cyclic
        # encoder functions for the dataset temporal resolution, inferred
        # from config["level"] (e.g. cm→monthly, cd→daily, cw→weekly).
        "use_cyclic_encoders": True,
    }

    return hyperparameters