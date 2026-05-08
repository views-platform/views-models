def get_sweep_config():
    """
    meow
    """
    sweep_config = {
        "method": "bayes",
        "name": "new_rules_nbeats_shadow_20260508_A",
        "early_terminate": {
            "type": "hyperband",
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
        "input_chunk_length": {"values": [36]},
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
        # WD range [2e-4, 1e-4, 5e-5]: LR floor ≈ 5e-4 × 0.5³ = 6e-5. No LayerNorm —
        # explicit WD is the primary regularizer against per-country basis memorization.
        # WD=2e-4 is 3.3× floor; θ_b basis vectors contract moderately, keeping outputs
        # from collapsing toward series mean. Upper bound: WD > 2e-4 collapses basis.
        "weight_decay": {"values": [2e-4, 1e-4, 5e-5]},
        # ==============================================================================
        # LR SCHEDULER: ReduceLROnPlateau
        # RLROP on val_loss: val_loss (test partition, frozen scalers) is significantly
        # smoother than train_loss on conflict batches, so RLROP's plateau detection
        # is reliable here. factor=0.5 (halve LR) is gentle enough for a noisy val
        # signal on ~200 series. patience=8: allows ~4 LR drops within the ESP=35
        # window (8, 16, 24, 32 epochs of stagnation) before early stopping triggers —
        # each drop gives the optimizer a fresh shot before committing to stop.
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["ReduceLROnPlateau"]},
        "lr_scheduler_factor": {"values": [0.5]},
        "lr_scheduler_patience": {"values": [12]},
        "lr_scheduler_min_lr": {"values": [1e-6]},
        "lr_scheduler_kwargs": {"values": [{"mode": "min", 
                                            "factor": 0.5, 
                                            "patience": 12, 
                                            "min_lr": 1e-6, 
                                            "threshold": 0.01, 
                                            "threshold_mode": "rel", 
                                            "cooldown": 3}]},
        # TiDE: skip path + unconstrained output → tight clipping. Pinned to
        # remove three-way interaction with weight_decay and dropout.
        # clip=5.0 removed: N-BEATS has no LayerNorm — 5.0 allows gradient spikes
        # that can blow through the FC stack without self-correction.
        "gradient_clip_val": {"values": [2.0, 3.0]},
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
        # N-BEATS ARCHITECTURE
        # ==============================================================================
        "generic_architecture": {"values": [True]},
        "num_stacks": {"values": [2, 3]},
        "num_blocks": {"values": [2, 3]},
        "num_layers": {"values": [3]},
        # layer_widths: FC width per block. Paper default is 512; 128/256 are
        # viable at our dataset scale (~14K windows). 512 included to cover
        # cases where conflict dynamics need wider representations.
        "layer_widths": {"values": [128, 256, 512]},
        # expansion_coefficient_dim: rank of the forecast basis projection.
        # Generic block: Linear(layer_width, ecd) → Linear(ecd, ocl=36).
        # ecd < ocl means the model can only express rank-ecd forecasts over
        # 36 steps. ecd=8/16 create a 4–8× bottleneck that is too restrictive
        # for multi-step conflict dynamics. Keep ecd >= ocl/2 at minimum.
        "expansion_coefficient_dim": {"values": [32, 64, 128]},
        "trend_polynomial_degree": {"values": [2]}, # useless for generic blocks but required by the rep gate
        # activation: ReLU is N-BEATS paper default.
        "activation": {"values": ["GELU"]},
        "use_reversible_instance_norm": {"values": [True]},
        "use_static_covariates": {"values": [True]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        # Dropout: N-BEATS is a deep MLP — moderate dropout needed for
        # ~200 series. Paper uses 0.0 but they had much more data.
        "dropout": {"values": [0.15, 0.25]},
        # ==============================================================================
        # LOSS FUNCTION: SpotlightLoss v36 (DRO)
        # ==============================================================================
        "loss_function": {"values": ["SpotlightLossLogcosh"]},
        "non_zero_threshold": {"values": [0.88]}, 
        # delta: multi-resolution spectral weight. DC bin masked.
        # "delta": {"distribution": "uniform", "min": 0.05, "max": 0.15},
        "delta": {"distribution": "uniform", "min": 0.0, "max": 0.1},
        # "static_covariate_stats": {"values": [{"transform": "AsinhTransform->MaxAbsScaler"}]},
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config
