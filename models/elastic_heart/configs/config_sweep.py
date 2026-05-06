def get_sweep_config():
    """
    """
    sweep_config = {
        "method": "bayes",
        "name": "elastic_heart_tsmixer_shadow_20260505_C",
        "early_terminate": {
            "type": "hyperband",
            # CAWR T_0=25: min_iter=30 = 5 epochs post-restart-1, past the spike; comparisons at matched post-restart phase.
            "min_iter": 30,
            "eta": 2,
        },
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        "input_chunk_length": {"values": [36, 48]},
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
        "lr": {"values": [5e-4, 2e-4, 1e-4]},
        # WD range [1e-4, 5e-5]: LR floor ≈ 5e-4 × 0.5³ = 6e-5. WD=1e-4 is 1.7× floor —
        # mild AdamW shrinkage; LayerNorm self-corrects scale drift. WD=0 removes
        # decoupled regularization entirely, risking per-country memorization.
        "weight_decay": {"values": [1e-4, 5e-5]},
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
        # clip=5.0 removed: TSMixer with GELU + sparse inputs can spike at 5.0.
        "gradient_clip_val": {"values": [1.5, 2.0, 3.0]},
        # ==============================================================================
        # SCALING
        # ==============================================================================
        "feature_scaler": {"values": [None]},
        "target_scaler": {"values": ["AsinhTransform"]},
        "feature_scaler_map": {
            "values": [{
                # Group 1: Zero-Anchor Preservation (Conflict & Heavy Macro)
                # Asinh compresses tails; MaxAbs scales to [-1, 1] keeping 0 at 0.
                "AsinhTransform->MaxAbsScaler": [
                    "lr_splag_1_ged_sb", "lr_splag_1_ged_ns", "lr_splag_1_ged_os",
                    "lr_ged_ns", "lr_ged_os",
                    "lr_ged_sb_delta", "lr_ged_ns_delta", "lr_ged_os_delta",
                    "lr_acled_sb", "lr_acled_sb_count", "lr_acled_os",
                    
                    "lr_wdi_ny_gdp_mktp_kd", "lr_wdi_nv_agr_totl_kn",
                    "lr_wdi_sm_pop_refg_or", "lr_wdi_sm_pop_netm",
                    "lr_wdi_dt_oda_odat_pc_zs",
                    "lr_wdi_ms_mil_xpnd_gd_zs",

                    "lr_vdem_v2x_horacc", "lr_vdem_v2x_veracc", "lr_vdem_v2x_diagacc",
                    "lr_vdem_v2xnp_client", "lr_vdem_v2xnp_regcorr",
                    "lr_vdem_v2xpe_exlpol", "lr_vdem_v2xpe_exlgeo",
                    "lr_vdem_v2xpe_exlgender", "lr_vdem_v2xpe_exlsocgr",
                    "lr_vdem_v2x_divparctrl", "lr_vdem_v2x_ex_party",
                    "lr_vdem_v2x_ex_military", "lr_vdem_v2x_genpp",
                    "lr_vdem_v2xeg_eqdr", "lr_vdem_v2xcl_prpty",
                    "lr_vdem_v2xeg_eqprotec", "lr_vdem_v2xcl_dmove",
                    "lr_vdem_v2x_clphy",

                    "lr_wdi_sp_pop_grow",          # signed, zero is meaningful inflection

                    "lr_wdi_sl_tlf_totl_fe_zs",    # bounded positive, no meaningful zero → [0,1]
                    "lr_wdi_se_enr_prim_fm_zs",    
                    "lr_wdi_sp_urb_totl_in_zs",    

                    "lr_wdi_sp_dyn_imrt_fe_in",   # Infant mortality
                    "lr_wdi_sh_sta_stnt_zs",      # Stunting
                    "lr_wdi_sh_sta_maln_zs",      # Malnutrition
                ],
            }],
        },
            
        # ==============================================================================
        # TSMIXER ARCHITECTURE
        # ==============================================================================
        "num_blocks": {"values": [2, 3]},
        "hidden_size": {"values": [128, 256, 512]},
        "ff_size": {"values": [128, 192]},
        "normalize_before": {"values": [True]},
        "activation": {"values": ["GELU"]},
        "norm_type": {"values": ["LayerNorm"]},
        
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        "dropout": {"values": [0.15, 0.25, 0.35]},
        "use_static_covariates": {"values": [True]},
        "use_reversible_instance_norm": {"values": [True]},
        
        # ==============================================================================
        # STATIC COVARIATE STATS
        # ==============================================================================
        # Per-entity fingerprint stats (mu, sigma, max, trend, sparsity) are
        # injected as static covariates into every TSMixer block via feature_mixing_static.
        # AsinhTransform alone leaves Syria mu≈5.3 vs peaceful countries at 0 — this
        # persistent 5× gap is injected at every block, biasing predictions upward
        # for high-conflict countries and causing systematic overprediction in the
        # 5–50 death range. MaxAbsScaler maps to [0,1]: Syria=1.0, peace=~0,
        # preserving relative order with no structural positive push.
        # Unlike TFT (VSN+GRN can learn to gate/rescale), TSMixer uses blunt linear
        # concatenation — cross-entity scale normalization must be explicit.
        "static_covariate_stats": {"values": [{"transform": "AsinhTransform->MaxAbsScaler"}]},
        
        # ==============================================================================
        # LOSS FUNCTION: SpotlightLoss
        # ==============================================================================
        "loss_function": {"values": ["SpotlightLossLogcosh"]},
        "non_zero_threshold": {"values": [0.88]}, 
        # delta: multi-resolution spectral weight. DC bin masked.
        "delta": {"distribution": "uniform", "min": 0.0, "max": 0.1},
        
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config