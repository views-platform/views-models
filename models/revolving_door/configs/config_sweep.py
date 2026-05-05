
def get_sweep_config():
    """meow"""

    sweep_config = {
        "method": "bayes",
        "name": "revolving_door_nhits_shadow_20260504_D",
        "early_terminate": {"type": "hyperband", "min_iter": 30, "eta": 2},
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        "input_chunk_length": {"values": [36, 48]},
        "output_chunk_length": {"values": [36]},
        "output_chunk_shift": {"values": [0]},
        "random_state": {"values": [67]},
        "mc_dropout": {"values": [False]},
        "optimizer_cls": {"values": ["AdamW"]},
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
        # explicit WD is the primary regularizer. WD=2e-4 is 3.3× floor (acceptable
        # for MLP stacks without self-correction). Wide range needed: N-HiTS pooling
        # stack benefits from stronger WD on deep runs to prevent per-country overfitting.
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
        # clip=5.0 removed: N-HiTS has no LayerNorm — 5.0 allows gradient spikes
        # that the architecture cannot self-correct.
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
                ],
                
                # Group 2: V-Dem IRT indices — already [0,1] by construction.
                # PassThrough keeps them at native scale, commensurate with
                # MaxAbsScaler's [-1, 1] output. StandardScaler would amplify
                # to ±2.5 (σ≈0.2), drowning conflict signal in feature-mixing layers.
                "PassThrough": [
                    "lr_vdem_v2x_horacc", "lr_vdem_v2x_veracc", "lr_vdem_v2x_diagacc",
                    "lr_vdem_v2xnp_client", "lr_vdem_v2xnp_regcorr",
                    "lr_vdem_v2xpe_exlpol", "lr_vdem_v2xpe_exlgeo",
                    "lr_vdem_v2xpe_exlgender", "lr_vdem_v2xpe_exlsocgr",
                    "lr_vdem_v2x_divparctrl", "lr_vdem_v2x_ex_party",
                    "lr_vdem_v2x_ex_military", "lr_vdem_v2x_genpp",
                    "lr_vdem_v2xeg_eqdr", "lr_vdem_v2xcl_prpty",
                    "lr_vdem_v2xeg_eqprotec", "lr_vdem_v2xcl_dmove",
                    "lr_vdem_v2x_clphy",
                ],
                # Group 3: Tail-Bounded Compression (Strictly Positive Indicators)
                # Asinh handles skew; MinMax utilizes full [0, 1] range since min >> 0.
                "MaxAbsScaler": [
                    "lr_wdi_sp_pop_grow",          # signed, zero is meaningful inflection
                ],
                "StandardScaler->MinMaxScaler": [
                    "lr_wdi_sl_tlf_totl_fe_zs",    # bounded positive, no meaningful zero → [0,1]
                    "lr_wdi_se_enr_prim_fm_zs",    
                    "lr_wdi_sp_urb_totl_in_zs",    
                ],
                # Group 4: Skewed health indicators — Asinh squashes tails, MinMax maps to [0,1]
                "AsinhTransform->MinMaxScaler": [
                    "lr_wdi_sp_dyn_imrt_fe_in",   # Infant mortality
                    "lr_wdi_sh_sta_stnt_zs",      # Stunting
                    "lr_wdi_sh_sta_maln_zs",      # Malnutrition
                ],
            }],
        },
        # ==============================================================================
        # N-HiTS ARCHITECTURE
        # ==============================================================================
        "num_stacks": {"values": [3]},
        "pooling_kernel_sizes": {"values": [[[4],[2],[1]]]},
        "n_freq_downsample": {"values": [[[3],[2],[1]]]},
        "max_pool_1d": {"values": [False]}, 
        "activation": {"values": ["GELU"]},
        "num_blocks": {"values": [1]},
        "num_layers": {"values": [3, 4]},
        "layer_widths": {"values": [256, 512]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        "dropout": {"values": [0.15, 0.25]},
        "use_static_covariates": {"values": [True]},
        # RevIN on: SpotlightLoss+AsinhTransform keeps outputs bounded; RevIN normalises
        # per-series mean/variance before encoding, improving convergence across heterogeneous
        # conflict intensities (peaceful vs. high-casualty series).
        "use_reversible_instance_norm": {"values": [True]},
        # ==============================================================================
        # LOSS FUNCTION: SpotlightLoss
        # ==============================================================================
        "loss_function": {"values": ["SpotlightLossLogcosh"]},
        "non_zero_threshold": {"values": [0.88]}, 
        # delta: multi-resolution spectral weight. DC bin masked.
        "delta": {"distribution": "uniform", "min": 0.0, "max": 0.1},
        "static_covariate_stats": {"values": [{"transform": "AsinhTransform->MaxAbsScaler"}]},
        # ModelCatalog builds the encoder dict from this flag at model-build
        # time, selecting functions based on config["level"] — JSON-safe.
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config