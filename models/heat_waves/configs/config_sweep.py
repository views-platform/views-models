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
        "name": "heat_waves_tft_shadow_202670506_A",
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
        # wd=0 removed: L2=0 allows unbounded weight growth at hidden=128 (600K params).
        # Sparse signal constraint: only ~24% of series are non-zero (event series).
        # At batch=128, ~32 effective gradient windows per step. Effective param density:
        # h=128: 600K / 13.5K event windows = 44 params/window — needs strong L2.
        # 1e-3 is the ceiling; 5e-5 is too light for this sparsity regime.
        "weight_decay": {"values": [1e-3, 5e-4, 1e-4]},
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
        "gradient_clip_val": {"values": [2.0, 3.0]},
        # ==============================================================================
        # SCALING
        # ==============================================================================
        "feature_scaler": {"values": [None]},
        "target_scaler": {"values": ["AsinhTransform"]},  # asinh(x): SpotlightLoss operates in asinh space. non_zero_threshold=0.88=asinh(1).
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
        # TFT ARCHITECTURE
        # ==============================================================================
        # hidden_size: Controls VSN, GRN, LSTM, and attention dimensions.
        # Sparse signal constraint: ~24% non-zero series = ~13.5K effective event windows.
        # hidden=256: 2.4M params / 13.5K event windows = 178 params/window → memorises
        #   the few conflict episodes (Syria 2012, Iraq 2014) instead of generalising.
        # hidden=128: 600K params / 13.5K event windows = 44 params/window — viable with
        #   strong regularisation (weight_decay≥1e-4, dropout≥0.25).
        # hidden=64: VSN GRN W∈ℝ^{64×64} too tight for 47-feature routing. Removed.
        "hidden_size": {"values": [128]},
        # lstm_layers: 1 layer with RevIN normalisation is optimal for sparse data.
        # Layer 2 would train on "hidden representation of ~75% zeros" — it learns
        # to suppress everything rather than extract conflict dynamics. Removed.
        "lstm_layers": {"values": [1]},
        # num_attention_heads: 2 heads at h=128 → head_dim=64. Wider heads are
        # better for sparse spike detection: each head attends to a 64-dim subspace
        # vs 4 heads at 32-dim. 32-dim attention heads may not resolve rare conflict
        # onset patterns from the dominant all-zero background.
        "num_attention_heads": {"values": [2]},
        "full_attention": {"values": [True]},
        "feed_forward": {"values": ["GatedResidualNetwork"]},
        # hidden_continuous_size: 32 is h/4 for h=128. With sparse data, most
        # continuous features are near-constant or zero for peaceful series —
        # the GRN doesn't need 64 dimensions to route a predominantly zero-valued
        # feature matrix. Smaller projection reduces overfit to structural covariates.
        "hidden_continuous_size": {"values": [32]},
        # categorical_embedding_sizes: empty dict for pure continuous features.
        "categorical_embedding_sizes": {"values": [{}]},
        # add_relative_index: Injects position information into attention.
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
        "dropout": {"values": [0.25, 0.35]},
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
        # SpotlightLoss pointwise gradient = w×tanh(e_shape) ≤ 2. Spectral bounded
        # at tanh ≤ 1. Ratio is ~2:1 — delta=0.10 gives spectral ~33% of total.
        # Particularly valuable for TFT — LSTM encoder has no explicit frequency bias.
        # min=0.05 enforces a spectral floor: delta=0 disables spectral loss entirely,
        # which for TFT (no frequency inductive bias) guarantees smooth forecasts.
        "delta": {
            "distribution": "uniform",
            "min": 0.05,
            "max": 0.15,
        },
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config