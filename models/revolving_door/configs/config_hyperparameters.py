def get_hp_config():
    """
    N-HiTS hyperparameters from v2 SpotlightLoss sweep best run.

    Returns:
    - hyperparameters (dict): Training configuration dictionary.
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
        "n_epochs": 300,
        "early_stopping_patience": 40,
        "early_stopping_min_delta": 0.0001,
        "force_reset": True,

        # Optimizer
        "optimizer_cls": "AdamW",
        "lr": 0.00012167435464868012,
        "weight_decay": 1e-3,
        "gradient_clip_val": 2.0,

        # LR Scheduler
        "lr_scheduler_cls": "CosineAnnealingWarmRestarts",
        "lr_scheduler_T_0": 30,
        "lr_scheduler_T_mult": 2,
        "lr_scheduler_eta_min": 1e-6,
        "lr_scheduler_kwargs": {
            "T_0": 30,
            "T_mult": 2,
            "eta_min": 1e-6,
        },
        "optimizer_kwargs": {
            "lr": 0.00012167435464868012,
            "weight_decay": 1e-3,
        },

        # Loss: SpotlightLoss (scale-invariant: Huber on relative error)
        "loss_function": "SpotlightLoss",
        "alpha": 0.504,
        "beta": 0.388,
        "kappa": 9.45,
        "delta": 0.502,
        "gamma": 0.0,

        # Scaling
        "feature_scaler": None,
        "target_scaler": "AsinhTransform",
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
                "lr_wdi_sl_tlf_totl_fe_zs",
                "lr_wdi_se_enr_prim_fm_zs",
                "lr_wdi_sp_urb_totl_in_zs",
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
            ],
        },

        # N-HiTS Architecture
        # 3 stacks: coarse (3:1), medium (2:1), fine (1:1) — limits interpolation leverage.
        # Average pooling + higher dropout suppress outlier-driven extrapolation.
        "num_stacks": 3,
        "num_blocks": 1,
        "num_layers": 2,
        "layer_widths": 128,
        "pooling_kernel_sizes": [[6], [3], [1]],
        "n_freq_downsample": [[6], [3], [1]],
        "max_pool_1d": False,
        "activation": "ReLU",
        "dropout": 0.25,
        "use_static_covariates": True,
        "use_reversible_instance_norm": False,

        # Temporal Encodings
        "add_encoders": {
            "position": {"past": ["relative"], "future": ["relative"]},
        },
    }

    return hyperparameters