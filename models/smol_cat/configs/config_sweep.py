def get_sweep_config():
    """
    meow
    """
    sweep_config = {
        "method": "bayes",
        "name": "smol_cat_tide_spotlight_v29_cm_msle",
        "early_terminate": {"type": "hyperband", "min_iter": 30, "eta": 2},
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        "input_chunk_length": {"values": [36, 48, 72]},
        "output_chunk_shift": {"values": [0]},
        "random_state": {"values": [67]},
        "output_chunk_length": {"values": [36]},
        "optimizer_cls": {"values": ["AdamW", "RAdam"]},
        "mc_dropout": {"values": [False]},
        "num_samples": {"values": [1]},
        "n_jobs": {"values": [-1]},
        # ==============================================================================
        # TRAINING
        # ==============================================================================
        # Batch size: Fixed at 64. MAAT magnitude weighting + hurdle produces
        # heterogeneous per-sample gradients — need enough samples for stable
        # batch gradient estimates. Country-level has fewer series than PGM,
        # so 64 is appropriate.
        "batch_size": {"values": [64]},
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [50]},
        "early_stopping_min_delta": {"values": [0.0001]},
        "force_reset": {"values": [True]},
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        # LR: MAAT has more gradient sources (4 components) — slightly wider
        # range to accommodate different component balance regimes.
        "lr": {
            "distribution": "log_uniform_values",
            "min": 5e-5,
            "max": 1e-3,
        },
        "weight_decay": {"values": [0, 1e-5, 1e-4]},
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["CosineAnnealingWarmRestarts"]},
        "lr_scheduler_T_0": {"values": [30]},
        "lr_scheduler_T_mult": {"values": [2]},
        "lr_scheduler_eta_min": {"values": [1e-6]},
        # MAAT: cosh weight is capped at w_max (default 100), and Huber base
        # limits gradient growth. Lower clip than raw JATLoss needed.
        "gradient_clip_val": {"values": [2.0, 3.0]},
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
        # TiDE ARCHITECTURE
        # ==============================================================================
        # Country-month: fewer series (~200) but richer temporal structure.
        # Need sufficient capacity to model diverse country trajectories.
        "num_encoder_layers": {"values": [1, 2]},
        # Decoder layers: 2 or 3. Country series are smoother than PGM —
        # 2 layers may suffice, but 3 gives more capacity for diverse patterns.
        "num_decoder_layers": {"values": [2, 3]},
        # decoder_output_dim: Dimensionality of the decoder output before
        # the temporal decoder. 32-64 is typical; 16 is the Darts default.
        "decoder_output_dim": {"values": [32, 64]},
        # hidden_size: SWEPT. Country-level needs capacity for ~200 diverse
        # trajectories. 256 is minimum viable, 512 gives headroom.
        "hidden_size": {"values": [256, 512]},
        # temporal_width_past: Width of past covariate projection output.
        # 0 bypasses projection (uses raw features). 4 is paper default,
        # 12 matches annual cycle for monthly data.
        "temporal_width_past": {"values": [4, 12]},
        # temporal_width_future: Width of future covariate projection output.
        # Larger values capture richer future covariate interactions.
        "temporal_width_future": {"values": [36, 48]},
        # temporal_decoder_hidden: Width of the temporal decoder MLP.
        # Needs enough capacity to map decoder output to final predictions.
        "temporal_decoder_hidden": {"values": [128, 256]},
        # temporal_hidden_size_past: Hidden layer width in past covariate
        # projection ResBlock. Defaults to hidden_size if None, which is
        # often too large. 64-128 is more appropriate.
        "temporal_hidden_size_past": {"values": [64, 128]},
        # temporal_hidden_size_future: Hidden layer width in future covariate
        # projection ResBlock. Same reasoning as past.
        "temporal_hidden_size_future": {"values": [64, 128]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        "use_layer_norm": {"values": [True, False]},
        # Dropout: Country-level has fewer training windows per series.
        # Slightly higher dropout ceiling to prevent overfitting on ~200 series.
        "dropout": {"values": [0.15, 0.25, 0.35]},
        "use_static_covariates": {"values": [True]},
        "use_reversible_instance_norm": {"values": [False, True]},
        # ==============================================================================
        # LOSS FUNCTION: PrismLoss
        # ==============================================================================
        # Four-component loss for asinh-transformed zero-inflated data:
        #   A. Magnitude-recovering weighted Huber (cosh Jacobian weight)
        #   B1. CDF temporal alignment (Cramér distance on cumulative sums)
        #   B2. Temporal derivative penalty (cosine-sim on first-differences)
        #   C. Asymmetric soft-focal classification (hurdle)
        #
        # Stability: w_max caps Jacobian weight. 99.9th-percentile per-sample
        # clamp inside loss. gradient_clip_val ≥ 1.0 externally.
        "loss_function": {"values": ["PrismLoss"]},
        # ── alpha (truth-only spotlight scale) ───────────────────────────────────────
        # 1+log_cosh(alpha*|y|) — truncated-inverse-density weight (Liu & Lin 2022;
        # Yang et al. 2021 LDS). No pred-side weight — gradient bounded by w(y)×tanh.
        # Weight at max UCDP (asinh≈11.5):
        #   alpha=0.15 → ≈2.1×   alpha=0.25 → ≈3.2×   alpha=0.35 → ≈4.3×
        # GRADIENT BUDGET: alpha scales pointwise gradient magnitude. Capped at 0.35
        # (4.3× max weight) so the pointwise-to-spectral gradient ratio stays in
        # [2:1, 6:1] across the full delta range. alpha=0.5 was 6.1× — starved
        # spectral of gradient budget at low delta, causing it to be ignored.
        # Test run anchor: alpha=0.2, delta=0.15 → balanced.
        "alpha": {
            "distribution": "uniform",
            "min": 0.15,
            "max": 0.35,
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
        # Floor at 0.08 so spectral is never noise. Cap at 0.25 so pointwise
        # accuracy isn't starved — the model still needs to get cell values right.
        "delta": {
            "distribution": "uniform",
            "min": 0.08,
            "max": 0.25,
        },
        # ── event_weight (balanced mean event/peace ratio) ────────────────────────────
        # Fraction of gradient budget allocated to event cells in balanced mean.
        # 0.50 = old 50/50 split (overpredicts). 0.25 = moderate. 0.10 = natural.
        "event_weight": {
            "distribution": "uniform",
            "min": 0.10,
            "max": 0.50,
        },
        "dual_mean": {"values": [True, False]},
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config