def get_sweep_config():
    """
    meow
    """
    sweep_config = {
        "method": "bayes",
        "name": "good_life_transformer_spotlight_v13_dcac_revin",
        "early_terminate": {"type": "hyperband", "min_iter": 50, "eta": 3},  # Rungs at 50,150,450 — 67% killed each rung → ~11% survive to rung 1. eta=3 safe: tight 3-dim loss space.
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        "input_chunk_length": {"values": [36]},
        "output_chunk_length": {"values": [36]},
        "output_chunk_shift": {"values": [0]},
        "random_state": {"values": [67]},
        "mc_dropout": {"values": [False]},
        "detect_anomaly": {"values": [False]},
        "optimizer_cls": {"values": ["AdamW"]},
        "num_samples": {"values": [1]},
        "n_jobs": {"values": [-1]},
        # ==============================================================================
        # TRAINING
        # ==============================================================================
        "batch_size": {"values": [64]},
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [40]},  # > T_0=30 (survives CAWR restart spike); stalled runs exit ~epoch 70-90 before rung 1 at 150
        "early_stopping_min_delta": {"values": [0.0001]},
        "force_reset": {"values": [True]},
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        # Transformers are more LR-sensitive than MLPs. Anchor ~3e-4 sits at ~80th
        # percentile on log scale of [1e-5, 5e-4] — conservative upper end while
        # still giving Bayes room to explore the lower half.
        "lr": {
            "distribution": "log_uniform_values",
            "min": 5e-5,
            "max": 5e-4,
        },
        "weight_decay": {"values": [0, 1e-4]},
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["CosineAnnealingWarmRestarts"]},
        "lr_scheduler_T_0": {"values": [30]},
        "lr_scheduler_T_mult": {"values": [2]},
        "lr_scheduler_eta_min": {"values": [1e-6]},
        # Max per-cell gradient = w(y)×tanh ≤ 4.3 (alpha=0.35). clip=5.0 never fires
        # and was removed. clip=3.0 trims only the most extreme event-cell spikes.
        "gradient_clip_val": {"values": [10.0]},
        # ==============================================================================
        # SCALING
        # ==============================================================================
        "feature_scaler": {"values": [None]},
        "target_scaler": {"values": ["AsinhTransform"]},
        "feature_scaler_map": {
            "values": [
                {
                    "AsinhTransform": [
                        "lr_wdi_sm_pop_refg_or",
                        "lr_wdi_ny_gdp_mktp_kd",
                        "lr_wdi_nv_agr_totl_kn",
                        "lr_splag_1_ged_sb",
                        "lr_splag_1_ged_ns",
                        "lr_splag_1_ged_os",
                        "lr_ged_ns", 
                        "lr_ged_os",
                    ],
                    "StandardScaler": [
                        "lr_ged_sb_delta",
                        "lr_ged_ns_delta",
                        "lr_ged_os_delta",
                        "lr_wdi_sm_pop_netm",
                        "lr_wdi_dt_oda_odat_pc_zs",
                        "lr_wdi_sp_pop_grow",
                        "lr_wdi_ms_mil_xpnd_gd_zs",
                        "lr_wdi_sp_dyn_imrt_fe_in",
                        "lr_wdi_sh_sta_stnt_zs",
                        "lr_wdi_sh_sta_maln_zs",
                    ],
                    "MinMaxScaler": [
                        "lr_wdi_sl_tlf_totl_fe_zs",
                        "lr_wdi_se_enr_prim_fm_zs",
                        "lr_wdi_sp_urb_totl_in_zs",
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
                }
            ]
        },
        # ==============================================================================
        # TRANSFORMER ARCHITECTURE
        # ==============================================================================
        # d_model: Embedding dimension. Constrained jointly with nhead so
        # that head_dim = d_model / nhead >= 32 for stable attention.
        # d_model=64, nhead=2 → head_dim=32 (minimum stable).
        # d_model=128, nhead=2 → head_dim=64 (comfortable).
        # nhead=4 excluded: valid for d_model=128 but gives head_dim=16 for
        # d_model=64, which is unstable — W&B can't condition on d_model.
        "d_model": {"values": [64, 128]},
        # nhead=2 valid for both d_model values (head_dim ≥ 32 in both cases).
        "nhead": {"values": [2]},
        # num_encoder_layers: 2-3 layers. ~200 series don't need deep
        # encoders; 2 is standard, 3 adds capacity for temporal complexity.
        "num_encoder_layers": {"values": [2, 3]},
        # num_decoder_layers: Match or slightly fewer than encoder.
        # Decoder complexity should mirror encoder for balanced attention.
        "num_decoder_layers": {"values": [2]},
        # dim_feedforward: FF expansion factor. 4× d_model raw, ~2× effective
        # with SwiGLU gating. 256=4× d_model=64, 512=4× d_model=128.
        # Bayes will naturally pair 64→256 and 128→512.
        "dim_feedforward": {"values": [256, 512]},
        # activation: Gated activations (GEGLU, SwiGLU) outperform vanilla
        # relu/gelu in recent Transformer literature (Shazeer 2020).
        "activation": {"values": ["SwiGLU"]},
        # norm_type: LayerNorm is standard and most stable.
        "norm_type": {"values": ["LayerNorm"]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        # Dropout: Transformers with ~200 series overfit fast. 0.15 is the
        # practical floor — below that, attention memorizes training windows.
        "dropout": {"values": [0.15, 0.25]},
        # use_reversible_instance_norm: RevIN on. v25 DC/AC eliminates the DC
        # bias that RevIN's σ multiplication used to amplify. No directional
        # bias to amplify → RevIN is purely beneficial (zero-mean residuals).
        "use_reversible_instance_norm": {"values": [True]},
        # ==============================================================================
        # LOSS FUNCTION: SpotlightLoss
        # ==============================================================================
        "loss_function": {"values": ["SpotlightLoss"]},
        # ── alpha (importance weight scale) ────────────────────────────────────────
        # 1+log_cosh(alpha*max(|y|,|ŷ_sg|)) — within-bucket magnitude priority.
        # v25 DC/AC: shape gradients sum to zero per series, so alpha only
        # steepens within-class priority without directional bias. Safe to go
        # higher than v12's 0.40 cap.
        # Weight at max UCDP (asinh≈11.5):
        #   alpha=0.15 → ≈1.9×   alpha=0.30 → ≈3.8×   alpha=0.50 → ≈6.2×
        # Floor at 0.15: mild priority — 1-death events get w≈1.01.
        # Cap at 0.50: zero-sum constraint means no runaway even at high alpha.
        "alpha": {
            "distribution": "uniform",
            "min": 0.15,
            "max": 0.50,
        },
        "non_zero_threshold": {"values": [0.88]},  # asinh(1) ≈ 0.88, i.e. ≥1 battle-related death
        # ── delta (multi-resolution spectral weight) ─────────────────────────────────
        # Spectral L1-magnitude matching (n_fft=6,12,24). Phase-insensitive by
        # the Fourier shift theorem: onset 1-mo early → ~zero spectral penalty.
        # n_fft=12 bin 1 = 12-month annual cycle — directly penalises missing seasonality.
        # n_fft=24 catches slow monotonic drift (smooth hockey sticks TV couldn't detect).
        # GRADIENT BUDGET: STFT accumulates ~48 gradient paths per time step across
        # 3 resolutions (8+14+26 bins×frames) vs 1 for pointwise. After .mean()
        # normalisation, spectral gradient norm is ~5-10× pointwise before delta.
        #   delta=0.08 → spectral ≈10-15% of total gradient (light regularisation)
        #   delta=0.15 → spectral ≈20-30% of total gradient (test run anchor)
        #   delta=0.25 → spectral ≈35-45% of total gradient (heavy temporal shaping)
        # Floor at 0.05 so spectral is never noise. Cap at 0.20 so pointwise
        # accuracy isn't starved — the model still needs to get cell values right.
        "delta": {
            "distribution": "uniform",
            "min": 0.05,
            "max": 0.20,
        },
        # ── event_weight (balanced mean event/peace ratio) ────────────────────────
        # Fraction of gradient budget allocated to event cells in balanced mean.
        # v25 DC/AC: shape loss gradients sum to zero per series by construction,
        # so event_weight no longer creates directional bias — it purely controls
        # how much gradient budget goes to event-cell shape accuracy vs peace.
        # Higher ew = more emphasis on getting war trajectories right.
        # Level calibration handled independently by the level-matching term.
        # Range [0.10, 0.40]: at 0.10 = natural prevalence (no class boost),
        # at 0.40 = 4× class boost for event shape — safe because zero-sum.
        "event_weight": {
            "distribution": "uniform",
            "min": 0.10,
            "max": 0.50,
        },
        # ── dual_mean ─────────────────────────────────────────────────────────────────
        # True = event/peace balanced mean. event_weight controls the class budget
        # ratio. v25 DC/AC eliminates DC offset structurally, but dual_mean still
        # provides cleaner gradient allocation — explicit class budget control
        # rather than implicit via importance weight distribution.
        "dual_mean": {"values": [True]},
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config