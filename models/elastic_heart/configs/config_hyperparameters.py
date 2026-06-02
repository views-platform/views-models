
def get_hp_config():
    """
    TSMixer hyperparameters
    Best run: elastic_heart_tsmixer_shadow_20260508_C
    lr=1e-4, clip=20, dropout=0.35, hidden=64, patience=15, RevIN=True
    """
    # r8
    hyperparameters = {
        # Temporal
        "steps": [*range(1, 36 + 1, 1)],
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
        "early_stopping_patience": 20,
        "early_stopping_min_delta": 0.001,
        "force_reset": True,

        # Optimizer
        "optimizer_cls": "AdamW",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "gradient_clip_val": 200.0, 

        # LR Scheduler
        "lr_scheduler_cls": "CosineAnnealingWarmRestarts",
        "lr_scheduler_T_0": 10,
        "lr_scheduler_T_mult": 2,
        "lr_scheduler_eta_min": 1e-6,
        "lr_scheduler_kwargs": {},
        "optimizer_kwargs": {
            "lr": 1e-3,
            "weight_decay": 1e-4,
        },
        "checkpoint_mode": "best",
        "loss_function": "SpotlightLossLogcosh",
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

        # TSMixer Architecture
        "num_blocks": 3,
        "hidden_size": 256,
        "ff_size": 512,
        "activation": "GELU",
        "norm_type": "LayerNorm",
        "normalize_before": True,
        "dropout": 0.25,
        "use_static_covariates": True,
        "use_reversible_instance_norm": True,

        # "static_covariate_stats": {
        #     "transform": "AsinhTransform->MaxAbsScaler",
        #     "inject": True,
        #     # "stats": ["trend", "sparsity"],
        # },

        "use_cyclic_encoders": True,
    }
    return hyperparameters
