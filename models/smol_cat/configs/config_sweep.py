def get_sweep_config():
    """
    meow
    """
    sweep_config = {
        "method": "bayes",
        "name": "smol_cat_tide_shadow_20260504",
        "early_terminate": {"type": "hyperband", "min_iter": 25, "eta": 2},
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
        "batch_size": {"values": [64]},
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
        # 1e-4 was negligible (5e-8/step). Pinned to reduce sweep dimensions.
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
        "lr_scheduler_factor": {"values": [0.7]},
        "lr_scheduler_patience": {"values": [10]},
        "lr_scheduler_min_lr": {"values": [1e-6]},
        "lr_scheduler_kwargs": {"values": [{"mode": "min", 
                                            "factor": 0.7, 
                                            "patience": 10, 
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
        "target_scaler": {"values": ["AsinhTransform"]},  # log1p(x): log-compresses targets, expm1 inverse
        "feature_scaler_map": {
            "values": [{
                "AsinhTransform": [
                    # Heavy-tailed counts, spatial lags, deltas: zero-inflated,
                    # 2–5 orders of magnitude cross-country range. asinh compresses
                    # the tail before StandardScaler centres. Handles negatives
                    # (deltas, net migration) as an odd function.
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
                "AsinhTransform->StandardScaler": [
                    # Macro volumes: 5+ order-of-magnitude cross-country difference.
                    # StandardScaler alone produces 50σ activations for large economies.
                    "lr_wdi_ny_gdp_mktp_kd",
                    "lr_wdi_nv_agr_totl_kn",
                    # Zero-inflated with heavy right tail.
                    "lr_wdi_sm_pop_refg_or",
                    "lr_wdi_dt_oda_odat_pc_zs",
                    # Signed, heavy tails both directions.
                    "lr_wdi_sm_pop_netm",
                    # Military % GDP: median ~1.5%, outliers at 10–25% (Saudi, NK).
                    # StandardScaler alone → 5–10σ activations for outlier countries.
                    "lr_wdi_ms_mil_xpnd_gd_zs",
                    # Infant mortality: Finland ~1.5, Chad ~90 — ~2 orders of magnitude.
                    # Strongly conflict-predictive; tail compression is essential.
                    "lr_wdi_sp_dyn_imrt_fe_in",
                    # Stunting/malnutrition 2–55%: asinh compresses 27× range to 3.3×.
                    # Right-skewed and conflict-predictive — tail signal matters.
                    "lr_wdi_sh_sta_stnt_zs",
                    "lr_wdi_sh_sta_maln_zs",
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
                "StandardScaler": [
                    # Rates and ratios without extreme skew or multi-order range.
                    # Pop growth: near-normal, signed. Female labour: bell-shaped ~35–50%.
                    # Enrolment ratio: clusters near 100. Urbanisation: near-uniform 10–90%.
                    "lr_wdi_sp_pop_grow",
                    "lr_wdi_sl_tlf_totl_fe_zs",
                    "lr_wdi_se_enr_prim_fm_zs",
                    "lr_wdi_sp_urb_totl_in_zs",
                ],
            }]
        },
        # ==============================================================================
        # TiDE ARCHITECTURE
        # ==============================================================================
        # Country-month: fewer series (~200) but richer temporal structure.
        # Need sufficient capacity to model diverse country trajectories.
        # num_encoder_layers: Das et al. (2023) TiDE paper uses N_e=2 as default.
        # Single encoder compresses the full (icl×features) representation in one
        # pass; a second layer creates a disentangled intermediate separating
        # structural country state (slow) from conflict onset signal (spiky/rare).
        # Bengio et al. (2013): deeper encoders produce more separable representations
        # for co-occurring patterns.
        # num_encoder_layers: Single layer compresses (icl×features) in one pass —
        # cannot disentangle structural country state from conflict onset signal.
        # 2–3 layers create an intermediate bottleneck where these separate.
        "num_encoder_layers": {"values": [2, 3]},
        # num_decoder_layers: Shallow decoders produce smooth mean-reversion outputs.
        # For ocl=36, generating conflict onsets (bursty) vs. sustained conflict
        # vs. peace-return requires 3 layers to shape the full output trajectory.
        "num_decoder_layers": {"values": [2, 3]},
        # decoder_output_dim: Dimensionality of the decoder output before
        # the temporal decoder. 32-64 is typical; 16 is the Darts default.
        # decoder_output_dim: Replicated across all ocl=36 output steps — temporal
        # decoder input is (decoder_output_dim × 36). At dim=128: 4,608-dim input
        # with ~200 training series → overfitting conflict-active window patterns.
        # Das et al. use 32 on larger datasets; 64 is the practical ceiling here.
        # decoder_output_dim: At dim=16, ocl=36: only 2.25 channels per timestep —
        # insufficient to simultaneously encode event presence and event shape.
        # 32 minimum viable; 64 gives headroom without overfitting (regularised
        # by temporal_decoder_hidden bottleneck + dropout + weight_decay).
        "decoder_output_dim": {"values": [32, 64]},
        # hidden_size: SWEPT. Country-level needs capacity for ~200 diverse
        # trajectories. 256 is minimum viable, 512 gives headroom.
        # 768 dropped: Kim et al. (2021) RevIN shows over-parameterized models in
        # heterogeneous multi-series settings collapse to the dominant mode (~95%
        # zeros). 512 is already generous; 768 adds parameters without additional
        # training series to constrain them.
        "hidden_size": {"values": [256, 512]},
        # temporal_width_past: Per-timestep projection of ~44 past features.
        # 12 forces a 3.7:1 compression per timestep; conflict splag/delta features
        # compete with structural features (V-Dem, WDI) for bandwidth. 24 doubles
        # per-timestep capacity into the encoder, preserving more conflict signal.
        # temporal_width_past: 44 features → 12 dims is 3.7:1 compression —
        # conflict splag/delta signals compete with structural features for bandwidth.
        # 24–36 preserves more conflict signal through the encoder.
        "temporal_width_past": {"values": [24]},
        # temporal_width_future: Future covariates = cyclic encoders only (~4 features).
        # Projecting 4 features to 8 dims adds noise, not signal. Match input dim.
        "temporal_width_future": {"values": [4, 8]},
        # temporal_decoder_hidden: MLP input = decoder_output_dim + temporal_width_future
        # = at most 64+8=72 dims. 512 is a 7× expansion of a 72-dim input with 200 series.
        # 128 is appropriate (1.8× expansion); 256 provides headroom for wider configs.
        "temporal_decoder_hidden": {"values": [128, 256]},
        # temporal_hidden_size_past: ResBlock hidden for ~44 past features → temporal_width_past.
        # 64 is a 1.5× expansion before compressing to 12-24; 128 provides more capacity.
        # Sweep both: Bayes will select per hidden_size config.
        "temporal_hidden_size_past": {"values": [64, 128]},
        # temporal_hidden_size_future: ResBlock hidden for ~4-8 cyclic future features.
        # 128 is a 16-32× expansion of 4-8 features — vastly over-parameterized.
        # Scale with feature count: 32-64 is appropriate.
        "temporal_hidden_size_future": {"values": [32]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        "use_layer_norm": {"values": [True]},
        # Dropout: Country-level has fewer training windows per series.
        # Slightly higher dropout ceiling to prevent overfitting on ~200 series.
        # dropout: Srivastava et al. (2014) optimal dropout scales with
        # parameters-per-series ratio. At hidden_size=512 with ~200 series,
        # 0.15 was tuned for larger benchmarks. Higher dropout prevents the
        # implicit "always predict zero" strategy by forcing the encoder to retain
        # conflict-relevant representations even in zero-dominated batches.
        # dropout: With Barron(1.5) providing stronger gradient flow than log_cosh,
        # lower dropout preserves conflict-onset gradient signal through the network.
        # 0.30 was tuned for log_cosh's saturating gradient; Barron tolerates less.
        "dropout": {"values": [0.20, 0.30]},
        "use_static_covariates": {"values": [True]},
        # RevIN on: SpotlightLoss DC/AC decomposition zeroes out per-series shape
        # gradients (Σ ∂L_shape/∂ŷᵢ = 0), preventing DC offset amplification through
        # RevIN denormalisation ŷ = ẑ·σ + μ. Safe even for sparse peace series.
        "use_reversible_instance_norm": {"values": [True]},
        "loss_function": {"values": ["SpotlightLoss"]},
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