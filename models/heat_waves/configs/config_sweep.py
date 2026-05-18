def get_sweep_config():
    """
    TFT + SpotlightLoss v36 sweep for country-month conflict forecasting.

    Architecture: TFT — LSTM encoder, multi-head attention, Variable Selection
    Networks, Gated Residual Networks. Static covariates (target_mu, target_sigma,
    target_max, target_trend, target_sparsity, country_id) flow through VSN → GRN
    gating, providing entity-conditioned temporal dynamics.

    SpotlightLoss v36 in asinh space. RevIN is safe here because the DC/AC
    decomposition ensures per-series shape gradients sum to exactly zero —
    RevIN denormalisation ŷ = ẑ·σ + μ cannot accumulate DC bias through
    the shape loss. non_zero_threshold=0.88 = asinh(1), i.e. ≥1 death.
    """
    sweep_config = {
        "method": "bayes",
        "name": "heat_waves_tft_shadow_20260508_D",
        "early_terminate": {"type": "hyperband", "min_iter": 25, "eta": 2},
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
        "time_steps": {"values": [36]},  # Checksum: Must match len(steps)
        "rolling_origin_stride": {"values": [1]},
        "prediction_format": {"values": ["dataframe"]},

        # ==============================================================================
        # TRAINING
        # ==============================================================================
        "batch_size": {"values": [128]},
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [30]},
        "early_stopping_min_delta": {"values": [1e-3]},
        "force_reset": {"values": [True]},
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        # 2e-4: TFT LSTM BPTT over 48 steps amplifies gradient depth — lower start
        # than TSMixer. RLROP halvings: 2e-4→1.4e-4→9.8e-5→6.9e-5 within ESP=30 budget.
        "lr": {"values": [5e-4, 2e-4]},
        # 1e-3 confirmed bad: crushes GRN/LSTM output weights → low-amplitude representation
        # that can't generalise sparse conflict peaks (val/train=2.92× observed at epoch 24
        # with wd=1e-3, event_bias=-21). Ceiling is 5e-4.
        "weight_decay": {"values": [5e-4, 2e-4, 1e-4]},
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        # ReduceLROnPlateau: loss-responsive, no cold-v̂ spike — avoids CAWR's
        # sudden peak hitting LSTM recurrent path unprepared.
        # factor=0.5: standard halving for LSTM BPTT (36-step gradient depth).
        # patience=10, cooldown=3: ~2-3 halvings within ESP=30.
        # threshold=0.01/rel: 1% relative improvement required — filters batch noise.
        "lr_scheduler_cls": {"values": ["ReduceLROnPlateau"]},
        "lr_scheduler_factor": {"values": [0.5]},
        "lr_scheduler_patience": {"values": [10]},
        "lr_scheduler_min_lr": {"values": [1e-6]},
        "lr_scheduler_kwargs": {"values": [{"mode": "min", 
                                            "factor": 0.5, 
                                            "patience": 10, 
                                            "min_lr": 1e-6, 
                                            "threshold": 0.01, 
                                            "threshold_mode": "rel", 
                                            "cooldown": 3}]},
        # TFT: LSTM BPTT over 36 steps accumulates gradient through the recurrent
        # path. GRN gating partially bounds this, but conflict-onset timesteps still
        # produce large gradients. Observed grad_norm/max=120 at clip=3 (97.5% cut).
        # No NaN/inf, no exploding layers → peaks are legitimate conflict-onset signal.
        # 7.0 upper bound: passes ~6% of peak vs 2.5% at clip=3 — meaningful improvement.
        "gradient_clip_val": {"values": [20.0, 50.0]},
        # ==============================================================================
        # SCALING
        # ==============================================================================
        "feature_scaler": {"values": [None]},
        "target_scaler": {"values": ["AsinhTransform"]},  # asinh(x): SpotlightLoss operates in asinh space. non_zero_threshold=0.88=asinh(1).
        "feature_scaler_map": {
            "values": [{
                # Group 1: Zero-Anchor Preservation (Conflict & Heavy Macro)
                # Asinh compresses tails; StandardScaler scales to zero mean and unit variance.
                "AsinhTransform->StandardScaler": [
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
            }],
        },
        # ==============================================================================
        # TFT ARCHITECTURE
        # ==============================================================================
        # hidden_size: Controls VSN, GRN, LSTM, and attention dimensions.
        # Sparse signal constraint: ~24% non-zero series = ~13.5K effective event windows.
        # hidden=256: 2.4M params / 13.5K event windows = 178 params/window → memorises
        #   the few conflict episodes (Syria 2012, Iraq 2014) instead of generalising.
        # hidden=128: 600K params / 13.5K event windows = 44 params/window — viable with
        #   strong regularisation (weight_decay≥1e-4, dropout≥0.25).
        # hidden=64: VSN GRN W∈ℝ^{64×64} too tight for 47-feature routing. Removed.
        "hidden_size": {"values": [64, 128]},
        # lstm_layers: pinned at 1. A second LSTM layer processes the hidden state
        # of the first — but after RevIN normalization the hidden representation is
        # dominated by peaceful series (90%). Layer 2 learns from a near-zero
        # distribution and degrades to a trivial suppressor that smooths the
        # already-averaged h_1. Single-layer LSTM with sufficient hidden_size is
        # more expressive for this sparsity regime.
        "lstm_layers": {"values": [1]},
        # num_attention_heads: pinned at 2 (head_dim=64 at h=128).
        # 4 heads → head_dim=32: each head attends to a narrower subspace; the
        # averaged output over 4 narrow heads smooths spike amplitude relative
        # to 2 wide heads. 64-dim heads better resolve rare conflict positions
        # against the dominant all-zero background.
        "num_attention_heads": {"values": [2]},
        # full_attention: critical for forecast smoothing.
        # full_attention=True (bidirectional decoder self-attention): step k attends
        # to all 36 output positions including future steps → implicit smoothing via
        # neighbour-borrowing. Reduces spike amplitude in output.
        # full_attention=False (causal): step k attends only to steps 1..k → each
        # step is forced to be independently resolved → less temporal smearing.
        # For conflict forecasting where sharpness matters, False is preferable.
        "full_attention": {"values": [False]},
        "feed_forward": {"values": ["GatedResidualNetwork"]},
        # hidden_continuous_size: 32 is h/4 for h=128. With sparse data, most
        # continuous features are near-constant or zero for peaceful series —
        # the GRN doesn't need 64 dimensions to route a predominantly zero-valued
        # feature matrix. Smaller projection reduces overfit to structural covariates.
        "hidden_continuous_size": {"values": [32]},
        # categorical_embedding_sizes: empty dict for pure continuous features.
        "categorical_embedding_sizes": {"values": [{}]},
        # add_relative_index: Pinned True. Without position encoding, attention
        # blends all output timesteps toward the batch mean — the model cannot
        # distinguish step 8 from step 20 and smooths spike amplitude across the
        # 36-step horizon. True is required for sharp temporal resolution.
        "add_relative_index": {"values": [True]},
        # skip_interpolation: Skip the interpolation in decoder output.
        "skip_interpolation": {"values": [True]},
        # norm_type: LayerNorm is standard and most stable for TFT.
        "norm_type": {"values": ["LayerNorm"]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        # Dropout: TFT applies dropout at 4 sites (LSTM, GRN, VSN, attention).
        # Compound survival (1-d)^4: 0.25 → 32% signal, 0.35 → 18% signal.
        # 0.15 removed: (1-0.15)^4=52% survival is insufficient for sparse data
        # (75% zero-target series). Model will memorise the ~13.5K event episodes.
        # 0.25-0.35 forces the LSTM to learn general conflict-onset patterns rather
        # than memorising specific country trajectories (Syria, Iraq, Ukraine).
        "dropout": {"values": [0.15, 0.25, 0.35]},
        "use_static_covariates": {"values": [True]},
        # RevIN on: SpotlightLoss DC/AC decomposition zeroes out per-series shape
        # gradients (Σ ∂L_shape/∂ŷᵢ = 0), preventing DC offset amplification through
        # RevIN denormalisation ŷ = ẑ·σ + μ. Also helps per-entity differentiation —
        # each entity's input window is normalized individually before LSTM encoding.
        "use_reversible_instance_norm": {"values": [True]},
        # ==============================================================================
        # LOSS FUNCTION: SpotlightLoss v36 (DRO)
        # ==============================================================================
        # asinh space + DC/AC decomposition (RevIN-safe) + compound weighting
        # (difficulty × importance, parameter-free) + KL-DRO log-space aggregation
        # + level anchor + multi-resolution spectral regularisation.
        "loss_function": {"values": ["SpotlightLossLogcosh"]},
        "non_zero_threshold": {"values": [0.88]},  # asinh(1) ≈ 0.88, i.e. ≥1 battle-related death
        # ── delta (multi-resolution spectral weight) ─────────────────────────────────
        # Spectral log_cosh(|S_pred| - |S_true|) at n_fft=6,12,24. DC bin masked.
        # min=0.00: allows Bayes to explore the full range of delta values.
        # max=0.20: at delta=0.20, spectral is ~25% of gradient — strong enough to
        # maintain spike sharpness without destabilizing the level anchor.
        # min=0.05: guarantees AC sharpness pressure in every run. delta=0 removes
        # the only mechanism that directly penalizes loss of spike amplitude in
        # frequency space, allowing the shape loss (smooth log_cosh) to converge
        # to a mean-regressing solution.
        "delta": {
            "distribution": "uniform",
            "min": 0.05,
            "max": 0.20,
        },
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        "use_cyclic_encoders": {"values": [False]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config