def get_sweep_config():
    """
    meow
    """
    sweep_config = {
        "method": "bayes",
        "name": "good_life_transformer_spotlight_v4_msle",
        "early_terminate": {"type": "hyperband", "min_iter": 50, "eta": 2},  # 50 > CAWR T_0=30 — avoids terminating runs at the LR spike before they recover
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
        "detect_anomaly": {"values": [False]},
        "optimizer_cls": {"values": ["AdamW"]},
        "num_samples": {"values": [1]},
        "n_jobs": {"values": [-1]},
        # ==============================================================================
        # TRAINING
        # ==============================================================================
        "batch_size": {"values": [64]},
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [50]},
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
            "min": 1e-5,
            "max": 5e-4,
        },
        "weight_decay": {"values": [1e-4]},
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["CosineAnnealingWarmRestarts"]},
        "lr_scheduler_T_0": {"values": [30]},
        "lr_scheduler_T_mult": {"values": [2]},
        "lr_scheduler_eta_min": {"values": [1e-6]},
        # Max per-cell gradient = w(y)×tanh ≤ 4.3 (alpha=0.35). clip=5.0 never fires
        # and was removed. clip=3.0 trims only the most extreme event-cell spikes.
        "gradient_clip_val": {"values": [2.0]},
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
        # 128/4=32 is the proven minimum; 128/2=64 gives richer heads.
        "d_model": {"values": [128]},
        # nhead: 4 gives head_dim=32 (tight but stable), 2 gives 64 (rich).
        # Both valid with d_model=128. Avoids the 64/4=16 trap entirely.
        "nhead": {"values": [4]},
        # num_encoder_layers: 2-3 layers. ~200 series don't need deep
        # encoders; 2 is standard, 3 adds capacity for temporal complexity.
        "num_encoder_layers": {"values": [2, 3]},
        # num_decoder_layers: Match or slightly fewer than encoder.
        # Decoder complexity should mirror encoder for balanced attention.
        "num_decoder_layers": {"values": [2]},
        # dim_feedforward: FF expansion factor. 2-4x d_model.
        # 256-512 for d_model=64-128. Controls capacity of position-wise FF.
        "dim_feedforward": {"values": [512]},
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
        "dropout": {"values": [0.15, 0.25, 0.35]},
        # use_reversible_instance_norm: Fixed True. Country series span asinh≈0
        # (Liechtenstein) to asinh≈11 (Syria). Without RevIN, Q/K/V magnitudes
        # are dominated by high-conflict rows — attention collapses to attending
        # only to Syria/Iraq regardless of the query series. RevIN normalizes
        # each series to unit variance before the encoder, fixing this.
        "use_reversible_instance_norm": {"values": [True]},
        # ==============================================================================
        # LOSS FUNCTION: SpotlightLoss
        # ==============================================================================
        "loss_function": {"values": ["SpotlightLoss"]},
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
            "min": 0.10,
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
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config