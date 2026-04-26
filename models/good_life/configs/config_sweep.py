def get_sweep_config():
    """
    meow
    """
    sweep_config = {
        "method": "bayes",
        "name": "good_life_transformer_prism_v24_2",
        "early_terminate": {"type": "hyperband", "min_iter": 25, "eta": 3},  # Rungs at 25,75,225. No CAWR restart spike — ReduceLROnPlateau is monotone-decaying so early measurements are valid.
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
        "batch_size": {"values": [256]},  # 256 stable for Transformer at d_model=256; larger batches improve gradient estimates and reduce steps/epoch, benign with AdamW.
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [30]},  # ReduceLROnPlateau patience=15: floor = 2 full reduction cycles (30 epochs) before ES can fire; guarantees at least one meaningful lr decay is acted on.
        "early_stopping_min_delta": {"values": [0.0001]},
        "force_reset": {"values": [True]},
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        # ReduceLROnPlateau decays monotonically so NaN risk is confined to the
        # first few epochs at peak lr. gradient_clip_val=10 + weight_decay guard
        # the ceiling. Floor at 3e-4: below this, ReduceLROnPlateau can decay
        # to min_lr too quickly, leaving < 20 effective epochs at useful lr.
        "lr": {
            "distribution": "log_uniform_values",
            "min": 3e-4,
            "max": 1e-3,
        },
        "weight_decay": {"values": [0, 1e-4, 1e-3]},  # wd=0 removed: weight max norm grew 20→42 in 29 epochs at wd=1e-4; need active regularisation. 1e-3 provides 10× stronger pull toward origin.
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        # ReduceLROnPlateau: reduce lr by factor when train_loss plateaus.
        # patience=15: loss CV≈0.20 means patience=10 fired on noise — run 1
        # took 7 reductions in 76 epochs, reaching min_lr with 224 wasted epochs.
        # patience=15 requires a genuine 15-epoch plateau before reducing.
        # factor=0.5: halves lr each step instead of ÷3.3. Same number of
        # reductions to reach min_lr but more useful training at each level.
        # min_lr=1e-6: practical floor below which updates are noise.
        "lr_scheduler_cls": {"values": ["ReduceLROnPlateau"]},
        "lr_scheduler_factor": {"values": [0.5]},
        "lr_scheduler_patience": {"values": [15]},
        "lr_scheduler_min_lr": {"values": [1e-6]},
        # gradient_clip_val=10 was too permissive: per-tensor gradient norms
        # hit 9.98 while median collapsed to 1e-14 — extreme bimodality caused
        # weight max norm to grow 20→42 in 29 epochs → overflow → NaN.
        # clip=1.0 reduces max per-step weight change by 10×, throttling the
        # runaway layers while preserving gradient direction for healthy ones.
        "gradient_clip_val": {"values": [1.0]},
        # ==============================================================================
        # SCALING
        # ==============================================================================
        "feature_scaler": {"values": [None]},
        "target_scaler": {"values": ["LogTransform"]},  # log1p(x): model operates directly in MSLE space
        "feature_scaler_map": {
            "values": [
                {
                    "LogTransform": [
                        "lr_wdi_sp_dyn_imrt_fe_in",
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
                    ],
                    "MinMaxScaler": [
                        "lr_wdi_sh_sta_stnt_zs",
                        "lr_wdi_sh_sta_maln_zs",
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
        # d_model=256 fixed: runs 2/3/5 (all d_model=128) showed monotonically
        # worse MSLE as lr increased — signature of capacity ceiling, not lr problem.
        # d_model=256, nhead=2 → head_dim=128 ✓
        # d_model=256, nhead=4 → head_dim=64 ✓
        "d_model": {"values": [256]},
        "nhead": {"values": [2, 4]},
        "num_encoder_layers": {"values": [2]},
        "num_decoder_layers": {"values": [2]},
        # dim_feedforward: with SwiGLU (effective ff/2), only ff=1024 gives a
        # proper 2× expansion (effective 512 = 2×d_model=256). ff=512 is the
        # degenerate identity case (effective 256 = 1×d_model); run 1 confirmed
        # it still works, but ff=1024 is the untested higher-capacity pairing.
        "dim_feedforward": {"values": [512, 1024]},
        # activation: Gated activations (GEGLU, SwiGLU) outperform vanilla
        # relu/gelu in recent Transformer literature (Shazeer 2020).
        "activation": {"values": ["SwiGLU"]},
        # norm_type: LayerNorm is standard and most stable.
        "norm_type": {"values": ["LayerNorm", "RMSNorm"]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        # Dropout: Transformers with ~200 series overfit fast. 0.15 is the
        # practical floor — below that, attention memorizes training windows.
        "dropout": {"values": [0.15, 0.25]},
        # use_reversible_instance_norm: Fixed off. RevIN bifurcates the loss
        # surface — optimal lr/alpha/dropout differ in each branch, so Bayes
        # models the average of two surfaces and converges poorly on both.
        # log1p + LayerNorm already handles the scale range (~11 units vs
        # ~50k raw). Test RevIN separately after sweep converges.
        "use_reversible_instance_norm": {"values": [True]},
        # ==============================================================================
        # LOSS FUNCTION: PrismLoss
        # ==============================================================================
        "loss_function": {"values": ["PrismLoss"]},
        "non_zero_threshold": {"values": [0.693]},  # log1p(1) ≈ 0.693, i.e. ≥1 battle-related death
        # ── delta (multi-resolution spectral weight) ─────────────────────────────────
        # Spectral log_cosh(|S_pred| - |S_true|) at n_fft=6,12,24. Phase-insensitive.
        # n_fft=12 bin 1 = 12-month annual cycle. DC bin masked.
        #
        # GRADIENT BUDGET (v33/MSE): pointwise MSE gradients scale as e (up to ~11
        # for large event cells). Spectral log_cosh gradient bounded at tanh ≤ 1.
        # Ratio is ~5-10:1 pointwise/spectral before delta — spectral is naturally
        # subordinate and won't dominate at any reasonable delta value.
        #   delta=0.05 → very light temporal regularisation (~2-4% of gradient)
        #   delta=0.10 → light (~4-8%)
        #   delta=0.20 → moderate (~8-15%)
        # Floor at 0.05: below this, spectral is noise relative to MSE signal.
        # Cap at 0.20: spectral purpose is shape regularisation, not accuracy.
        "delta": {
            "distribution": "uniform",
            "min": 0.05,
            "max": 0.20,
        },
        # ── event_weight (balanced mean active/peace ratio) ────────────────────────
        # Only used when dual_mean=True.
        # With MSE, natural event gradient share is already ~50% (10% cells × 9×
        # squared error vs 90% cells × 1× squared error). event_weight above 0.25
        # amplifies events beyond their natural MSE dominance → overprediction risk.
        # v17 run (ew=0.20) still overpredicted (y_hat_bar=40). Keep upper end tight.
        "event_weight": {
            "values": [0.50], # not used
        },
        # ── dual_mean ─────────────────────────────────────────────────────────────────
        # v17 run showed train_loss=0.28 vs MSLE_val=0.80 — 3× gap caused by
        # balanced mean and MSLE having different fixed points. The model converged
        # to the balanced-mean minimum (grad_norm/max=0.051 at CAWR restart) which
        # is NOT the MSLE minimum. Sweep both to test: False makes training loss
        # = MSLE exactly, eliminating the objective mismatch.
        "dual_mean": {"values": [False]},
        # ── ohem_ratio (Online Hard Example Mining) ──────────────────────────────────
        # Fraction of cells kept (by hardest MSE). 1.0 = disabled (plain mean).
        # 0.3 = top 30%. Bypasses dual_mean — hard cells are overwhelmingly events.
        # Sweep both disabled (1.0) and moderate (0.3) to test impact.
        # "ohem_ratio": {"values": [1.0, 0.3, 0.2, 0.1]},
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config