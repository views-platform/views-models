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
        "name": "heat_waves_tft_spotlight_v1_10_rlrop",
        "early_terminate": {"type": "hyperband", "min_iter": 25, "eta": 2},
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
        "early_stopping_patience": {"values": [30]},
        "early_stopping_min_delta": {"values": [1e-3]},
        "force_reset": {"values": [True]},
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        # 2e-4: TFT LSTM BPTT over 48 steps amplifies gradient depth — lower start
        # than TSMixer. RLROP halvings: 2e-4→1.4e-4→9.8e-5→6.9e-5 within ESP=30 budget.
        "lr": {"values": [2e-4]},
        # wd=0 removed: L2=0 allows unbounded weight growth at hidden=256 (2.4M params).
        # 1e-4/1e-3 bracket the useful regularisation range.
        "weight_decay": {"values": [1e-4, 1e-3]},
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        # ReduceLROnPlateau: loss-responsive, no cold-v̂ spike — avoids CAWR's
        # sudden peak hitting LSTM recurrent path unprepared.
        # factor=0.7: gentle halving for LSTM BPTT (48-step gradient depth).
        # patience=10, cooldown=3: ~2-3 halvings within ESP=30.
        # threshold=0.01/rel: 1% relative improvement required — filters batch noise.
        "lr_scheduler_cls": {"values": ["ReduceLROnPlateau"]},
        "lr_scheduler_factor": {"values": [0.7]},
        "lr_scheduler_patience": {"values": [10]},
        "lr_scheduler_min_lr": {"values": [1e-6]},
        "lr_scheduler_kwargs": {"values": [{"mode": "min", "factor": 0.7, "patience": 10, "min_lr": 1e-6, "threshold": 0.01, "threshold_mode": "rel", "cooldown": 3}]},
        # SpotlightLoss v36: compound weight w ∈ [1,2), gradient = w×tanh(e) ≤ 2.
        # TFT LSTM compounds gradients through time — clip=5.0 never fires on
        # the loss itself but catches rare LSTM state explosions on outlier series.
        # 10.0 is too loose (lazy guard); 1s.0 too tight for LSTM backprop.
        "gradient_clip_val": {"values": [1.0, 3.0, 5.0]},
        # ==============================================================================
        # SCALING
        # ==============================================================================
        "feature_scaler": {"values": [None]},
        "target_scaler": {"values": ["AsinhTransform"]},  # asinh(x): SpotlightLoss operates in asinh space. non_zero_threshold=0.88=asinh(1).
        "feature_scaler_map": {
            "values": [{
                "AsinhTransform": [
                    # Heavy-tailed: conflict counts, GDP, refugees, ODA
                    "lr_splag_1_ged_sb", "lr_splag_1_ged_ns", "lr_splag_1_ged_os",
                    "lr_ged_ns", "lr_ged_os",
                    "lr_ged_sb_delta", "lr_ged_ns_delta", "lr_ged_os_delta",
                    "lr_wdi_ny_gdp_mktp_kd", "lr_wdi_nv_agr_totl_kn",
                    "lr_wdi_sm_pop_refg_or", "lr_wdi_dt_oda_odat_pc_zs",
                    "lr_wdi_sp_pop_grow", "lr_wdi_sp_urb_totl_in_zs",
                    "lr_wdi_sm_pop_netm",
                ],
                "MinMaxScaler": [
                    # Bounded [0,1] or near-bounded: V-Dem indices, WDI rates
                    "lr_vdem_v2x_horacc", "lr_vdem_v2x_veracc", "lr_vdem_v2x_diagacc",
                    "lr_vdem_v2xnp_client", "lr_vdem_v2xnp_regcorr",
                    "lr_vdem_v2xpe_exlpol", "lr_vdem_v2xpe_exlgeo",
                    "lr_vdem_v2xpe_exlgender", "lr_vdem_v2xpe_exlsocgr",
                    "lr_vdem_v2x_divparctrl", "lr_vdem_v2x_ex_party",
                    "lr_vdem_v2x_ex_military", "lr_vdem_v2x_genpp",
                    "lr_vdem_v2xeg_eqdr", "lr_vdem_v2xcl_prpty",
                    "lr_vdem_v2xeg_eqprotec", "lr_vdem_v2xcl_dmove",
                    "lr_vdem_v2x_clphy",
                    "lr_wdi_ms_mil_xpnd_gd_zs", "lr_wdi_sh_sta_stnt_zs",
                    "lr_wdi_sh_sta_maln_zs", "lr_wdi_sl_tlf_totl_fe_zs",
                    "lr_wdi_se_enr_prim_fm_zs", "lr_wdi_sp_dyn_imrt_fe_in",
                ],
            }]
        },
        # ==============================================================================
        # TFT ARCHITECTURE
        # ==============================================================================
        # hidden_size: Controls VSN, GRN, LSTM, and attention dimensions.
        # TFT is parameter-inefficient — hidden_size propagates into every
        # sub-module (LSTM 4×h², VSN features×h, GRN 2×h², attention 3×h²).
        # At hidden=64:  ~150K params — VSN bottleneck; head_dim=32 (minimum stable).
        # At hidden=128: ~600K params — viable; head_dim=64.
        # At hidden=256: ~2.4M params — capacity ceiling; head_dim=128. Needs
        #   dropout=0.25 + weight_decay=1e-4 to avoid memorisation (~54K windows).
        # VSN routing of 41 features is the binding constraint at h=64 — the GRN
        # bottleneck (W∈ℝ^{h×h}) barely separates conflict vs. structural feature
        # clusters. h=128/256 give it room. Bayes will pick the winner.
        "hidden_size": {"values": [64, 128, 256]},
        # lstm_layers: 1 layer must compress 48-month context to one hidden state
        # before attention. Conflict data has lag structures up to 12–24 months
        # (escalation cycles) — 2 layers allow a "memory" + "pattern" hierarchy.
        # Cost is modest (extra 4h² weights per layer).
        "lstm_layers": {"values": [1, 2]},
        # num_attention_heads: nhead=2 valid for all hidden sizes (head_dim≥32).
        # nhead=4 valid for hidden=128 (head_dim=32) and hidden=256 (head_dim=64)
        # but degenerate for hidden=64 (head_dim=16). Fixed at 2 to stay safe.
        "num_attention_heads": {"values": [2, 4]}, # 4 degenerates at hidden=64
        "full_attention": {"values": [True]},
        "feed_forward": {"values": ["GatedResidualNetwork"]},
        # hidden_continuous_size: Scale with hidden_size. 32=h/2 for h=64, 64=h/2 for h=128,
        # 128=h/2 for h=256. Bayes will pair appropriately.
        "hidden_continuous_size": {"values": [32, 64]},
        # categorical_embedding_sizes: empty dict for pure continuous features.
        "categorical_embedding_sizes": {"values": [{}]},
        # add_relative_index: Injects position information into attention.
        "add_relative_index": {"values": [True]},
        # skip_interpolation: Skip the interpolation in decoder output.
        "skip_interpolation": {"values": [False]},
        # norm_type: LayerNorm is standard and most stable for TFT.
        "norm_type": {"values": ["LayerNorm"]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        # Dropout: TFT applies dropout at 4 sites (LSTM, GRN, VSN, attention).
        # Compound survival (1-d)^4: 0.15 → 52% signal, 0.25 → 32% signal.
        # GRN gating adds implicit regularisation so explicit dropout stays moderate.
        "dropout": {"values": [0.15, 0.25]},
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
        "delta": {
            "distribution": "uniform",
            "min": 0.05,
            "max": 0.10,
        },
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config