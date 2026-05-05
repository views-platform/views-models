
def get_sweep_config():
    """meow"""

    sweep_config = {
        "method": "bayes",
        "name": "revolving_door_nhits_shadow_20260505_A",
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
        # patience=12: same reasoning as TCN — patience=8 exhausted the LR budget
        # by epoch ~36. patience=12 spaces halvings across the 300-epoch window.
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
        # N-HiTS ARCHITECTURE
        # ==============================================================================
        "num_stacks": {"values": [3]},
        "pooling_kernel_sizes": {"values": [[[4],[2],[1]]]},
        "n_freq_downsample": {"values": [[[3],[2],[1]]]},
        "max_pool_1d": {"values": [False, True]}, 
        "activation": {"values": ["GELU"]},
        # num_blocks: pinned to 1. pooling_kernel_sizes and n_freq_downsample
        # require inner tuples of length == num_blocks per stack — they cannot
        # be swept independently with Bayes. Capacity reduction is achieved
        # entirely through layer_widths and num_layers instead.
        "num_blocks": {"values": [1]},
        # num_layers: 3/4→2/3. Deeper per-block MLPs with wide layers is what
        # produces vanishing grad_norm/min ≈ 4e-13 in downstream stacks —
        # the signal is consumed by the time it reaches the earlier layers
        # during backprop. Shallower + narrower keeps gradients alive.
        "num_layers": {"values": [2, 3]},
        # layer_widths: 256/512→64/128. The 8.2M param count (178 params/sample)
        # is the root cause of the freeze: stack 0 absorbs the dominant zero
        # pattern completely in epoch 0, leaving near-zero residuals for stacks
        # 1 and 2. 64/128 brings total params to ~300K–1.2M (6–26 params/sample)
        # — still expressive but not able to memorise the distribution instantly.
        "layer_widths": {"values": [64, 128]},
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