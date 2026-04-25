def get_sweep_config():
    """
    meow
    """
    sweep_config = {
        "method": "bayes",
        "name": "heat_waves_tft_prism_v4_msle",
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
            "min": 5e-5,
            "max": 1e-3,
        },
        "weight_decay": {"values": [0, 1e-4]},
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
        "target_scaler": {"values": ["LogTransform"]},  # log1p(x): model operates in MSLE space. MSE in log1p = MSLE exactly.
        "feature_scaler_map": {
            "values": [
                {
                    "LogTransform": [
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
        # hidden_size: 64 is capacity-limited for conflict dynamics (head_dim=32 minimum
        # at nhead=2). 128 doubles capacity. Both valid — Bayes picks the winner.
        "hidden_size": {"values": [64, 128]},
        "lstm_layers": {"values": [1]},
        # num_attention_heads: nhead=2 valid for both hidden sizes (head_dim≥32).
        # nhead=4 valid for hidden=128 (head_dim=32) but invalid for hidden=64 (head_dim=16).
        # Fixed at 2 to avoid degenerate hidden=64/nhead=4 pairing.
        "num_attention_heads": {"values": [2]},
        "full_attention": {"values": [True]},
        "feed_forward": {"values": ["GatedResidualNetwork"]},
        # hidden_continuous_size: Scale with hidden_size. 32=half of 64, 64=half of 128.
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
        # RevIN off: log1p space, peace series have σ≈0 → RevIN divides by σ → NaN.
        # log1p + LayerNorm already handles scale range (~11 units vs ~50k raw).
        "use_reversible_instance_norm": {"values": [False]},
        # ==============================================================================
        # LOSS FUNCTION: PrismLoss
        # ==============================================================================
        "loss_function": {"values": ["PrismLoss"]},
        # alpha removed in PrismLoss v33. MSE in log1p = MSLE exactly.
        # log_cosh+alpha was causing 10-16× gradient deficit via tanh saturation.
        "non_zero_threshold": {"values": [0.693]},  # log1p(1) ≈ 0.693, i.e. ≥1 battle-related death
        # ── delta (multi-resolution spectral weight) ─────────────────────────────────
        # Spectral log_cosh(|S_pred| - |S_true|) at n_fft=6,12,24. DC bin masked.
        # MSE pointwise gradient scales as e (up to ~11). Spectral bounded at tanh ≤ 1.
        # Particularly valuable for TFT — LSTM encoder has no explicit frequency bias.
        "delta": {
            "distribution": "uniform",
            "min": 0.05,
            "max": 0.20,
        },
        # dual_mean=False: training loss = MSLE exactly. True caused false minimum.
        "dual_mean": {"values": [False]},
        "event_weight": {
            "values": [0.50], # not used
        },
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config