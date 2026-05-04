
def get_sweep_config():
    """meow"""

    sweep_config = {
        "method": "bayes",
        "name": "revolving_door_nhits_shadow_20260504",
        "early_terminate": {"type": "hyperband", "min_iter": 25, "eta": 2},
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        "input_chunk_length": {"values": [48]},
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
        # wd=1e-3: at lr=5e-4, effective wd/step = 5e-7 — sufficient for ~200 series.
        "weight_decay": {"values": [1e-3, 1e-4]},
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
        "gradient_clip_val": {"values": [1.0, 3.0, 5.0]},
        # ==============================================================================
        # SCALING
        # ==============================================================================
        "feature_scaler": {"values": [None]},
        "target_scaler": {"values": ["AsinhTransform"]},
        "feature_scaler_map": {
            "values": [{
                "AsinhTransform->StandardScaler": [
                    # Macro volumes: 5+ order-of-magnitude cross-country difference.
                    # StandardScaler alone produces 50σ activations for large economies.
                    "lr_wdi_ny_gdp_mktp_kd",
                    "lr_wdi_nv_agr_totl_kn",
                    # Zero-inflated with heavy right tail.
                    "lr_wdi_sm_pop_refg_or",
                
                    # Signed, heavy tails both directions.
                    "lr_wdi_sm_pop_netm",
                
                    # Infant mortality: Finland ~1.5, Chad ~90 — ~2 orders of magnitude.
                    # Strongly conflict-predictive; tail compression is essential.
                    "lr_wdi_sp_dyn_imrt_fe_in",
                    
                    # Rates and ratios without extreme skew or multi-order range.
                    # Pop growth: near-normal, signed.
                    "lr_wdi_sp_pop_grow",
                    # Female labour: bell-shaped ~35–50%.
                    # Enrolment ratio: clusters near 100. Urbanisation: near-uniform 10–90%.
                    "lr_wdi_sl_tlf_totl_fe_zs",
                    "lr_wdi_se_enr_prim_fm_zs",
                    "lr_wdi_sp_urb_totl_in_zs",

                    # Stunting/malnutrition 2–55%: asinh compresses 27× range to 3.3×.
                    # Right-skewed and conflict-predictive — tail signal matters.
                    "lr_wdi_sh_sta_stnt_zs",
                    "lr_wdi_sh_sta_maln_zs",
                    "lr_wdi_dt_oda_odat_pc_zs",

                    # Military % GDP: median ~1.5%, outliers at 10–25% (Saudi, NK).
                    # StandardScaler alone → 5–10σ activations for outlier countries.
                    "lr_wdi_ms_mil_xpnd_gd_zs",
                ],
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
                ],
                "MinMaxScaler": [
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
            }]
        },
        # ==============================================================================
        # N-HiTS ARCHITECTURE
        # ==============================================================================
        # Pooling kernel rationale (icl=48):
        #   kernel=4  → 48/4  = 12 quarterly groups (4 per annual cycle — cleanly
        #              resolves annual seasonality by Nyquist; pool=6 gave only 2.67
        #              points/cycle, barely Nyquist with misaligned phase)
        #   kernel=2  → 48/2  = 24 bi-monthly groups (sub-quarterly detail)
        #   kernel=1  → 48 raw monthly steps        (fine stack always unpooled)
        # n_freq_downsample rationale (ocl=36, independent of icl):
        #   fd=3 → 36/3 = 12 basis functions (trend + 5 harmonics + conflict patterns;
        #          fd=6 left only 4 slots after DC+trend — no room for conflict dynamics)
        #   fd=2 → 36/2 = 18 basis functions (sub-quarterly detail)
        #   fd=1 → 36/1 = 36 basis functions (full monthly)
        "num_stacks": {"values": [3]},
        "pooling_kernel_sizes": {"values": [[[4],[2],[1]]]},
        "n_freq_downsample": {"values": [[[3],[2],[1]]]},  # Darts requires fd[-1][-1]==1. Fine stack is full-rank; regularization via dropout/wd/width instead.
        "max_pool_1d": {"values": [False]},  # True causes spike propagation via max-pooling on sparse conflict data (confirmed Run 2)
        "activation": {"values": ["GELU"]},
        # num_blocks/num_layers/layer_widths: Darts defaults are 1 block, 2 layers,
        # 512 width per stack. num_blocks fixed at 1: zero-inflation is addressed by
        # SpotlightLoss+AsinhTransform, not within-stack depth. num_blocks=2 requires
        # paired pooling/downsampling tuples per stack and overfits sparse peace series.
        # Capacity is swept via layer_widths instead.
        # num_blocks fixed at 1: Bayes sweeps pick num_blocks and
        # pooling_kernel_sizes independently. num_blocks=2 requires paired tuples
        # per stack (e.g. [[6,6],[3,3],[1,1]]) — incompatible with [[6],[3],[1]].
        # Architecturally: within-stack depth is redundant here; SpotlightLoss +
        # AsinhTransform handle zero-inflation. Capacity is swept via layer_widths.
        "num_blocks": {"values": [1]},
        # num_layers: Pinned at 3. Width is the primary capacity knob; sweeping
        # both width and depth creates an interaction Bayes can't resolve efficiently.
        "num_layers": {"values": [3]},
        # layer_widths: 768 gives ~5M params for ~48K training samples (14:1 ratio)
        # → severe overfitting risk. 256–512 is the right range with RevIN fixing
        # the mean bias, reducing the capacity needed to correct overprediction.
        "layer_widths": {"values": [256, 512]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        "dropout": {"values": [0.20, 0.30]},
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
        "delta": {"distribution": "uniform", "min": 0.05, "max": 0.15},
        # ModelCatalog builds the encoder dict from this flag at model-build
        # time, selecting functions based on config["level"] — JSON-safe.
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config