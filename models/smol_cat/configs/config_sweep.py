def get_sweep_config():
    """
    TiDE Hyperparameter Sweep Configuration - JATLoss
    ==================================================

    Strategy: Asinh-space regression with Jacobian-asymmetric temporal weighting
    -----------------------------------------------------------------------------

    v21: JAT-Loss sweep — Jacobian-weighted magnitude + temporal alignment
    -----------------------------------------------------------------------
    Replaces FME-Loss with a simpler 3-component design:
      1. Jacobian weight cosh(max(ŷ,y)) — directly inverts asinh compression
      2. Asymmetric penalty β — FN costs (1+β)× more than FP
      3. Temporal Huber loss on first-differences — penalizes onset/offset errors

    Advantages over FME-Loss:
    - Fewer hyperparameters (4 vs 10) — smaller sweep space
    - Jacobian weight is mathematically exact (cosh = d/dy sinh)
    - No separate class-weight machinery — asymmetry via single β parameter
    - Temporal term directly addresses flatline by penalizing constant predictions

    Features: 37 (6 conflict + 13 WDI + 18 V-Dem, no topics).
    Input tensor per window: 37 × 36 = 1,332 values.
    Training windows: ~16.5K.

    Architecture notes (same as v17+):
    - RevIN disabled: Jacobian weighting handles magnitude natively
    - mc_dropout=False: Deterministic inference
    - Position encoder: relative only (integer month_id indexing)
    """
    sweep_config = {
        "method": "bayes",
        "name": "smol_cat_tide_jat_v21_msle",
        "early_terminate": {"type": "hyperband", "min_iter": 30, "eta": 2},
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        "input_chunk_length": {"values": [36]},  # FIX: match output_chunk_length. 48/60 adds noise with 37 features.
        "output_chunk_shift": {"values": [0]},
        "random_state": {"values": [67]},
        "output_chunk_length": {"values": [36]},
        "optimizer_cls": {"values": ["Adam"]},
        # mc_dropout=False: Deterministic inference. Dropout at inference
        # suppresses activations by (1-p), compounding signal loss across
        # decoder layers.
        "mc_dropout": {"values": [False]},
        "num_samples": {"values": [1]},
        "n_jobs": {"values": [-1]},
        # ==============================================================================
        # TRAINING
        # ==============================================================================
        # Batch size: Fixed at 64. Jacobian weighting on high-magnitude
        # samples can produce large per-sample gradients — need enough
        # samples per batch for stable gradient estimates.
        "batch_size": {"values": [64]},
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [40]},
        "early_stopping_min_delta": {"values": [0.0001]},
        "force_reset": {"values": [True]},
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        # LR is loss-dependent — always sweep. Narrowed range from prior sweeps.
        "lr": {
            "distribution": "log_uniform_values",
            "min": 5e-5,
            "max": 2e-4,
        },
        # Weight decay: Fixed at geometric mean of prior narrow range.
        # Negligible effect vs loss params.
        "weight_decay": {"values": [5e-6]},
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["CosineAnnealingWarmRestarts"]},
        "lr_scheduler_T_0": {"values": [30]},
        "lr_scheduler_T_mult": {"values": [2]},
        "lr_scheduler_eta_min": {"values": [1e-6]},
        # JAT-Loss: cosh weight is unbounded (cosh(9)≈4052). Fixed at 5.0 —
        # sufficient since JAT has fewer multiplicative amplifiers than FME.
        "gradient_clip_val": {"values": [1.0, 1.5, 3.0]},
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
        # TiDE ARCHITECTURE (narrowed — fix low-impact, sweep high-impact)
        # ==============================================================================
        # Encoder layers: Fixed at 1 (paper default). With 37 clean features,
        # the encoder's job is easy. Decoder needs the depth, not encoder.
        "num_encoder_layers": {"values": [1]},
        # Decoder layers: SWEPT. Directly affects anti-flatline. Must sustain
        # signal across 36 steps from a single latent.
        "num_decoder_layers": {"values": [2, 3]},
        # decoder_output_dim: Fixed at 64. 32 too small for 36-step output,
        # 128 overkill for 1-dim target.
        "decoder_output_dim": {"values": [64]},
        # hidden_size: SWEPT. Most impactful arch param. 128 dropped (too
        # compressed to sustain signal over 36 steps).
        "hidden_size": {"values": [256, 512]},
        # temporal_width_past: SWEPT. 24 dropped (too wide for 37 features,
        # averages noise). 4 (paper) vs 12 (1 year cycle).
        "temporal_width_past": {"values": [4, 12]},
        # temporal_width_future: SWEPT. 24 too short, 64 overkill.
        # 36 = horizon match, 48 = moderate context.
        "temporal_width_future": {"values": [36, 48]},
        # temporal_decoder_hidden: Fixed at 256. 128 underpowered for
        # 36-step decoder, 512 overkill for ~16K training windows.
        "temporal_decoder_hidden": {"values": [256]},
        # temporal_hidden_size_past/future: Fixed at 128. 3.5× input dim
        # is the right compression ratio for 37 features.
        "temporal_hidden_size_past": {"values": [128]},
        "temporal_hidden_size_future": {"values": [128]},
        # ==============================================================================
        # REGULARIZATION (narrowed — fix layer_norm, tighten dropout)
        # ==============================================================================
        # Layer norm: Fixed at True. Stabilizes Jacobian-weighted gradient flow.
        "use_layer_norm": {"values": [True]},
        # Dropout: SWEPT but narrowed. High dropout kills rare conflict
        # signal. Light regularization only.
        "dropout": {
            "distribution": "uniform",
            "min": 0.05,
            "max": 0.15,
        },
        "use_static_covariates": {"values": [True]},
        # RevIN disabled: Jacobian weighting (cosh) handles magnitude scaling
        # natively. RevIN's per-series normalization would distort the asinh
        # space that the Jacobian correction relies on.
        "use_reversible_instance_norm": {"values": [False]},
        # ==============================================================================
        # LOSS FUNCTION: JATLoss (Jacobian-Asymmetric Temporal Loss)
        # ==============================================================================
        # Three-component loss for asinh-transformed zero-inflated data:
        #   1. Jacobian weight cosh(max(ŷ,y)) — exact asinh compression correction
        #   2. Asymmetric penalty β           — FN costs (1+β)× more than FP
        #   3. Temporal Huber on Δŷ vs Δy     — penalizes onset/offset timing errors
        #
        # Stability: 99.9th-percentile per-sample clamp inside loss. Requires
        # gradient_clip_val ≥ 5.0 externally. batch_size ≥ 64 recommended.
        "loss_function": {"values": ["JATLoss"]},
        #
        # ── beta ──────────────────────────────────────
        # Asymmetric penalty for under-prediction (false negatives).
        # FN cost = (1+β) × FP cost. Primary FN/FP balance knob.
        #   0.3 = mild asymmetry (1.3× FN)
        #   0.5 = moderate (1.5× FN, recommended)
        #   1.0 = strong (2× FN)
        #   2.0 = aggressive (3× FN)
        "beta": {
            "distribution": "uniform",
            "min": 0.3,
            "max": 2.0,
        },
        # ── lambda_mag ────────────────────────────────
        # Weight for the magnitude component. Fixed at 1.0 as the reference
        # scale — lambda_time is tuned relative to this.
        "lambda_mag": {"values": [1.0]},
        # ── lambda_time ───────────────────────────────
        # Weight for temporal alignment component.
        # Too low: model ignores timing. Too high: overwhelms magnitude signal.
        #   0.01 = light temporal guidance
        #   0.1  = moderate (recommended starting point)
        #   0.5  = strong temporal constraint
        "lambda_time": {
            "distribution": "log_uniform_values",
            "min": 0.01,
            "max": 0.5,
        },
        # ── huber_kappa ───────────────────────────────
        # Huber delta for temporal gradient loss. Controls sensitivity to
        # sharp onsets vs gradual trends.
        #   0.5 = more robust to sharp spikes (treats large Δ as linear)
        #   1.0 = balanced (recommended)
        #   2.0 = closer to MSE on temporal gradients
        "huber_kappa": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 2.0,
        },
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        # NOTE: cyclic and datetime_attribute encoders require DatetimeIndex,
        # but VIEWS uses integer month_id indexing. Only position encoders
        # are compatible with integer-indexed TimeSeries.
        #
        # position:relative generates a [0.0, ..., 1.0] ramp over the window.
        # Darts position encoder only supports "relative" — "absolute" raises
        # ValueError. This is the only future signal beyond bare features.
        "add_encoders": {
            "values": [
                {
                    "position": {"past": ["relative"], "future": ["relative"]},
                }
            ]
        },
    }

    sweep_config["parameters"] = parameters
    return sweep_config