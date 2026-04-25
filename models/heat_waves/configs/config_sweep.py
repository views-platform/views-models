def get_sweep_config():
    """
    meow
    """
    sweep_config = {
        "method": "bayes",
        "name": "heat_waves_tft_spotlight_v3_msle",
        "early_terminate": {"type": "hyperband", "min_iter": 35, "eta": 2},  # 50 > CAWR T_0=30 — avoids terminating runs at the LR spike before they recover
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
        "batch_size": {"values": [64]},
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [50]},
        "early_stopping_min_delta": {"values": [0.0001]},
        "force_reset": {"values": [True]},
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        # TFT combines LSTM + attention — more complex gradient landscape
        # than pure Transformers. GRN gating dampens gradient magnitudes,
        # so TFT tolerates a slightly broader LR range than vanilla Transformer.
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
        # Max per-cell gradient = w(y)×tanh ≤ 4.3 (alpha=0.35). clip=5.0 never
        # fires; 3.0 barely fires. TFT has LSTM gradients that compound through
        # time — tighter clip prevents rare-event spikes from destabilising LSTM state.
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
        # TFT ARCHITECTURE
        # ==============================================================================
        # hidden_size: Controls VSN, GRN, LSTM, and attention dimensions.
        # TFT is parameter-inefficient — hidden_size propagates into every
        # sub-module (LSTM 4×h², VSN features×h, GRN 2×h², attention 3×h²).
        # At hidden=128: ~600K params on ~200 series = instant memorization.
        # At hidden=64: ~150K params — tight but viable for sparse data.
        # head_dim = hidden_size / nhead = 64/2 = 32 (minimum stable).
        "hidden_size": {"values": [64]},
        # lstm_layers: 1 is standard (Lim et al. 2021). Attention handles
        # long-range, LSTM provides local context. On 85% zero data, LSTM
        # converges to a "peace" attractor — additional layers just deepen
        # this attractor without adding new information.
        "lstm_layers": {"values": [1]},
        # num_attention_heads: 2 heads with hidden_size=64 → head_dim=32
        # (minimum stable softmax). Two attention patterns: "where was
        # conflict?" + "what changed structurally?" — sufficient for ~8
        # informative positions in a 48-step window.
        "num_attention_heads": {"values": [2]},
        # full_attention: True = O(n²) over full encoder+decoder context.
        # With icl=48 this is cheap (48²=2304). On sparse data, the rare
        # non-zero positions are by definition far apart — the decoder needs
        # full access to find them. Sparse attention risks masking the few
        # informative positions.
        "full_attention": {"values": [True]},
        # feed_forward: GatedResidualNetwork is TFT's native architecture.
        # GLU gating provides implicit regularization on zero inputs — gate
        # learns to close → skip connection dominates. A strength for sparse data.
        "feed_forward": {"values": ["GatedResidualNetwork"]},
        # hidden_continuous_size: Pre-VSN embedding dim for continuous features.
        # With hidden_size=64, 32 = 2× compression before VSN. Sufficient
        # projection dim — features carry most information in a few dimensions
        # (conflict counts + deltas).
        "hidden_continuous_size": {"values": [32]},
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
        # Dropout: TFT applies dropout in LSTM, GRN, VSN, and attention — 4
        # independent sites. Compound survival rate at dropout=d: (1-d)^4.
        #   0.10 → 66% signal survives    0.15 → 52%    0.25 → 32%
        # TFT's GRN gating already provides implicit regularization, so
        # explicit dropout can be lower than pure Transformer.
        "dropout": {"values": [0.10, 0.15, 0.25]},
        "use_static_covariates": {"values": [True]},
        # RevIN: Country series span asinh≈0 (Liechtenstein) to asinh≈11
        # (Syria). Without RevIN, LSTM hidden state magnitudes diverge across
        # scales → VSN gates open fully for high-conflict series, stay shut
        # for peaceful ones → OOD inputs get arbitrary outputs.
        "use_reversible_instance_norm": {"values": [True]},
        # ==============================================================================
        # LOSS FUNCTION: PrismLoss
        # ==============================================================================
        "loss_function": {"values": ["PrismLoss"]},
        # ── alpha (truth-only spotlight scale) ───────────────────────────────────────
        # 1+log_cosh(alpha*|y|) — truncated-inverse-density weight (Liu & Lin 2022;
        # Yang et al. 2021 LDS). No pred-side weight — gradient bounded by w(y)×tanh.
        # Weight at max UCDP (asinh≈11.5):
        #   alpha=0.15 → ≈2.1×   alpha=0.25 → ≈3.2×   alpha=0.35 → ≈4.3×
        # GRADIENT BUDGET: alpha scales pointwise gradient magnitude. Capped at 0.35
        # (4.3× max weight) so the pointwise-to-spectral gradient ratio stays in
        # [2:1, 6:1] across the full delta range.
        "alpha": {
            "distribution": "uniform",
            "min": 0.15,
            "max": 0.35,
        },
        "non_zero_threshold": {"values": [0.88]},  # asinh(1) ≈ 0.88, i.e. ≥1 battle-related death
        # ── delta (multi-resolution spectral weight) ─────────────────────────────────
        # Spectral L1-magnitude matching (n_fft=6,12,24). Phase-insensitive by
        # the Fourier shift theorem: onset 1-mo early → ~zero spectral penalty.
        # Particularly valuable for TFT because the LSTM encoder has no explicit
        # frequency bias — spectral loss is the only thing constraining its
        # temporal structure.
        #   delta=0.08 → spectral ≈10-15% of total gradient (light regularisation)
        #   delta=0.15 → spectral ≈20-30% of total gradient (test run anchor)
        #   delta=0.25 → spectral ≈35-45% of total gradient (heavy temporal shaping)
        "delta": {
            "distribution": "uniform",
            "min": 0.08,
            "max": 0.25,
        },
        # ── event_weight (balanced mean event/peace ratio) ────────────────────────────
        # Fraction of gradient budget allocated to event cells in balanced mean.
        "event_weight": {
            "distribution": "uniform",
            "min": 0.10,
            "max": 0.50,
        },
        # ── dual_mean ─────────────────────────────────────────────────────────────────
        # True = event/peace balanced mean (event_weight controls ratio).
        # False = plain per-cell mean (event_weight ignored).
        "dual_mean": {"values": [True, False]},
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config