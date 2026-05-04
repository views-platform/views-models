def get_sweep_config():
    """
    meow
    """
    sweep_config = {
        "method": "bayes",
        "name": "good_life_transformer_shadow_202605034_C",
        "early_terminate": {"type": "hyperband", "min_iter": 25, "eta": 2},  # Rungs at 30,90,270. min_iter=30 = 5 epochs post-restart-1, past the spike; comparisons at matched post-restart phase.
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        "input_chunk_length": {"values": [48]},
        "output_chunk_shift": {"values": [0]},
        "random_state": {"values": [67]},
        "output_chunk_length": {"values": [36]},
        "optimizer_cls": {"values": ["AdamW"]},
        "mc_dropout": {"values": [False]},
        "num_samples": {"values": [1]},
        "n_jobs": {"values": [-1]},
        # ==============================================================================
        # TRAINING
        # ==============================================================================
        "batch_size": {"values": [128]},
        "n_epochs": {"values": [300]},
        # ESP=35: allows ~4 LR reductions (patience=8 each) before triggering.
        # Each RLROP firing gives the optimizer a reset opportunity; 35 epochs of
        # continuous stagnation despite all reductions is a reliable stop signal.
        # Hyperband (min_iter=15) is the primary fast-kill for clearly bad runs.
        "early_stopping_patience": {"values": [35]},
        "early_stopping_min_delta": {"values": [0.001]},
        "force_reset": {"values": [True]},
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        "lr": {"values": [5e-4, 2e-4]},
        # wd=1e-3: at lr=5e-4, effective wd/step = 5e-7 — sufficient for ~200 series.
        "weight_decay": {"values": [1e-3, 1e-4]},
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
         "lr_scheduler_cls": {"values": ["ReduceLROnPlateau"]},
        "lr_scheduler_factor": {"values": [0.5]},
        "lr_scheduler_patience": {"values": [8]},
        "lr_scheduler_min_lr": {"values": [1e-6]},
        "lr_scheduler_kwargs": {"values": [{"mode": "min", 
                                            "factor": 0.5, 
                                            "patience": 8, 
                                            "min_lr": 1e-6, 
                                            "threshold": 0.01, 
                                            "threshold_mode": "rel", 
                                            "cooldown": 3}]},
        # TiDE: skip path + unconstrained output → tight clipping. Pinned to
        # remove three-way interaction with weight_decay and dropout.
        "gradient_clip_val": {"values": [1.0, 2.0, 3.0]},
        # ==============================================================================
        # SCALING
        # ==============================================================================
        "feature_scaler": {"values": [None]},
        "target_scaler": {"values": ["AsinhTransform"]},
        "feature_scaler_map": {
            "values": [{
                "AsinhTransform->MaxAbsScaler": [
                    # Conflict counts, spatial lags, deltas: zero-inflated,
                    # 2–5 orders of magnitude cross-country range. asinh compresses
                    # the tail; MaxAbsScaler maps to [0,1] preserving zero=0 anchor
                    # and full proportional tail discrimination (no mean-shift).
                    "lr_splag_1_ged_sb",
                    "lr_splag_1_ged_ns",
                    "lr_splag_1_ged_os",
                    "lr_ged_ns",
                    "lr_ged_os",
                    "lr_ged_sb_delta",
                    "lr_ged_ns_delta",
                    "lr_ged_os_delta",
                    "lr_acled_sb",
                    "lr_acled_sb_count",
                    "lr_acled_os",

                    # Macro volumes: 5+ order-of-magnitude cross-country difference.
                    # StandardScaler alone produces 50σ activations for large economies.
                    "lr_wdi_ny_gdp_mktp_kd",
                    "lr_wdi_nv_agr_totl_kn",
                    # Zero-inflated with heavy right tail.
                    "lr_wdi_sm_pop_refg_or",
                
                    # Signed, heavy tails both directions.
                    "lr_wdi_sm_pop_netm",
                
                    "lr_wdi_dt_oda_odat_pc_zs",

                    # Military % GDP: median ~1.5%, outliers at 10–25% (Saudi, NK).
                    # StandardScaler alone → 5–10σ activations for outlier countries.
                    "lr_wdi_ms_mil_xpnd_gd_zs",
                ],
                "PassThrough": [
                    # V-Dem [0,1] IRT indices: IRT construction places empirical range
                    # near the full [0,1] interval across ~200 countries. Many are
                    # bimodal or heavily skewed (e.g. v2x_ex_military: most near 0,
                    # some near 1). StandardScaler destroys this structure; MinMaxScaler
                    # maps to [0,1] matching the index construction.
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
                "StandardScaler": [
                    # Zero is a meaningful inflection (growth vs contraction): mean-centering
                    # preserves directional contrast for channel-mixing MLPs.
                    # sp_pop_grow is signed; MinMaxScaler squashes −2% and +2% symmetrically
                    # away from 0, destroying the contraction-vs-expansion signal.
                    "lr_wdi_sp_pop_grow",
                    # Bounded positives (labour, enrolment, urbanisation): StandardScaler
                    # centres the distribution — MLP distinguishes low/high by sign of z-score.
                    "lr_wdi_sl_tlf_totl_fe_zs",
                    "lr_wdi_se_enr_prim_fm_zs",
                    "lr_wdi_sp_urb_totl_in_zs",
                ],
                "AsinhTransform->MinMaxScaler": [
                    # Right-skewed ratios with no meaningful zero: asinh compresses
                    # the tail; MinMaxScaler maps empirical [min, max] → [0, 1].
                    # MaxAbsScaler wastes bottom range when min >> 0.
                    "lr_wdi_sp_dyn_imrt_fe_in",   # infant mortality [1.5, 90]
                    "lr_wdi_sh_sta_stnt_zs",       # stunting % [2, 55]
                    "lr_wdi_sh_sta_maln_zs",       # malnutrition % [2, 45]
                ],
            }]
        },
        # ==============================================================================
        # TRANSFORMER ARCHITECTURE
        # ==============================================================================
        "d_model": {"values": [128]}, 
        "nhead": {"values": [4]},
        "num_encoder_layers": {"values": [2, 3, 4]},
        "dim_feedforward": {"values": [256, 512]},
        "activation": {"values": ["GELU"]},
        "norm_type": {"values": ["LayerNorm", "RMSNorm"]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        "dropout": {"values": [0.15, 0.25]},
        "use_reversible_instance_norm": {"values": [True]},
        # ==============================================================================
        # LOSS FUNCTION: SpotlightLossLogcosh
        # ==============================================================================
        "loss_function": {"values": ["SpotlightLossLogcosh"]},
        "non_zero_threshold": {"values": [0.88]}, 
        # delta: multi-resolution spectral weight. DC bin masked.
        "delta": {"distribution": "uniform", "min": 0.05, "max": 0.15},
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config