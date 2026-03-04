def get_sweep_config():
    """
    TiDE Hyperparameter Sweep Configuration - FocalMagnitudeExpandingLoss
    =====================================================================

    Strategy: Asinh-space regression with focal magnitude-expanding weighting
    --------------------------------------------------------------------------

    v17: Anti-flatline sweep — no topics (output_chunk_length=36 fixed)
    -------------------------------------------------------------------
    Root cause of flatline: TiDE's decoder generates all 36 steps in one
    forward pass from a single latent vector + position encoding. The position
    encoding (linear ramp) carries no conflict-relevant signal, so the decoder
    learns to predict the unconditional mean (≈0) for steps beyond ~5.

    Features: 37 (6 conflict + 13 WDI + 18 V-Dem, no topics).
    Input tensor per window: 37 × 36 = 1,332 values.
    Training windows: ~16.5K.

    Fixes applied (without changing output_chunk_length):

    1. RevIN swept [True, False]: Per-series denormalization rescales outputs
       back to each country's historical range, preventing Sudan from collapsing
       to the global near-zero mean. Primary anti-flatline mechanism.

    2. Topics excluded entirely: Noisy topic features (64→16→0) were the
       majority of inputs but contributed weakest signal. With 37 high-quality
       features, the encoder has a cleaner compression task.

    3. Architecture right-sized for 37 features:
       - hidden_size [128, 256, 512]: 128 viable now (37→128 = 3.5:1 compression)
       - temporal_hidden_size_past [64, 128, 256]: 64 added (proportional to input dim)
       - num_encoder_layers [1, 2] (paper default: 1)
       - decoder_output_dim [32, 64, 128] (paper default: 32)
       - temporal_width_past swept [4, 12, 24] (paper default: 4)

    4. Deeper decoder [2, 3]: More capacity to sustain signal across 36 steps.

    5. temporal_decoder_hidden [128, 256, 512]: More capacity per output step.

    6. mc_dropout=False: Deterministic inference eliminates dropout-induced
       signal suppression.

    7. Position encoder: relative only (Darts doesn't support absolute).
       VIEWS uses integer month_id indexing, so cyclic/datetime encoders
       are also incompatible.
    """
    sweep_config = {
        "method": "bayes",
        "name": "smol_cat_tide_fmel_v19_msle",
        "early_terminate": {"type": "hyperband", "min_iter": 30, "eta": 2},
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        "input_chunk_length": {"values": [36, 48, 60]},
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
        # Batch size: FME-Loss stability notes require ≥64 (need enough
        # conflict samples per batch for stable gradient estimates when
        # ~85% of targets are zero). 32 is too noisy with three unbounded
        # multipliers.
        "batch_size": {"values": [64, 128]},
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [40]},
        "early_stopping_min_delta": {"values": [0.0001]},
        "force_reset": {"values": [True]},
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        # Widened for Charbonnier: different gradient profile than Huber
        # (non-saturating tails) may shift the optimal LR.
        "lr": {
            "distribution": "log_uniform_values",
            "min": 3e-5,
            "max": 3e-4,
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 2e-6,
            "max": 8e-6,
        },
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["CosineAnnealingWarmRestarts"]},
        "lr_scheduler_T_0": {"values": [30]},
        "lr_scheduler_T_mult": {"values": [2]},
        "lr_scheduler_eta_min": {"values": [1e-6]},
        # FME-Loss has three unbounded multipliers (Charbonnier gradient ×
        # inflation × exp magnitude). Stability notes require clip ≥ 5.0.
        # Clipping at 1.0 would re-introduce the gradient ceiling the loss
        # is designed to avoid. 5.0 is the safe floor; 10.0 gives more
        # headroom for large-event signal.
        "gradient_clip_val": {"values": [5.0, 10.0]},
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
        # TiDE ARCHITECTURE
        # ==============================================================================
        # Encoder layers: Paper uses 1 for most benchmarks. With only 37
        # high-quality features, 1 layer is likely sufficient.
        "num_encoder_layers": {"values": [1, 2]},
        # Decoder layers: More critical than encoder for anti-flatline.
        # Must sustain signal across 36 steps from a single latent.
        # 2-3 layers give the decoder nonlinear capacity to learn
        # position-dependent output transformations.
        "num_decoder_layers": {"values": [2, 3]},
        # decoder_output_dim: Paper default is 32. This is the per-step
        # dimensionality before the final projection to 1. With a 1-dim
        # target and noisy inputs, 256 is way overkill — the model has
        # more output capacity than useful signal. 32 (paper) to 128.
        "decoder_output_dim": {"values": [32, 64, 128]},
        # hidden_size: The main encoder/decoder bottleneck. With 37 features
        # (input tensor 37×36=1332), 128 gives a 10:1 compression — viable
        # now that noisy topics are gone. 256 worked in prior sweeps.
        # 512 for anti-flatline (richer latent at distant steps).
        "hidden_size": {"values": [128, 256, 512]},
        # temporal_width_past: Paper default is 4. With noisy features,
        # very wide receptive field (24) may average noise into the signal.
        # Swept to let Bayes find whether narrow (4, paper) or wide (24,
        # previous) works better. 12 is the compromise for monthly data
        # where meaningful trends span ~1 year.
        "temporal_width_past": {"values": [4, 12, 24]},
        # temporal_width_future: Controls how much of the output horizon
        # each position can see. Larger values help maintain coherent
        # predictions by allowing cross-position information flow.
        # 24 is the new floor (2/3 of horizon). 48 and 64 give full
        # horizon context.
        # 36 added: matches output_chunk_length exactly (natural landmark).
        "temporal_width_future": {"values": [24, 36, 48, 64]},
        # temporal_decoder_hidden: MLP width in temporal decoder.
        # Paper default is 128. 256 is reasonable. 512 may overfit
        # given ~16K training windows. Include 128 (paper) as floor.
        "temporal_decoder_hidden": {"values": [128, 256, 512]},
        # temporal_hidden_size_past: Processes per-timestep features.
        # With 37 features, 64 = ~1.7× input dim (reasonable floor).
        # 128 and 256 for more capacity.
        "temporal_hidden_size_past": {"values": [64, 128, 256]},
        "temporal_hidden_size_future": {"values": [128, 256]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        "use_layer_norm": {"values": [True, False]},
        # Floor lowered to 0.05: with 95% zero targets, rare non-zero
        # activations are fragile — high dropout further suppresses them.
        # TiDE paper default is 0.1. Let Bayes explore light regularization.
        "dropout": {
            "distribution": "uniform",
            "min": 0.05,
            "max": 0.25,
        },
        "use_static_covariates": {"values": [True]},
        # Without autoregressive chaining, RevIN is the primary mechanism to
        # prevent collapse to the global unconditional mean.
        #
        # How it helps: RevIN normalizes each series to zero-mean/unit-var at
        # input, then REVERSES the normalization at output. So even if the
        # decoder predicts a flat "0.3" for all steps in normalized space,
        # the denormalization rescales it to Sudan's range (~500+) and
        # Norway's range (~0). The decoder only needs to learn relative
        # temporal patterns, not absolute magnitudes.
        #
        # Risk: Zeros become negative in normalized space, distorting the
        # zero_threshold boundary. But the flatline is worse than threshold
        # distortion — a threshold can be recalibrated, a flatline cannot.
        #
        # Sweep both to let Bayes measure the tradeoff.
        "use_reversible_instance_norm": {"values": [False]},
        # ==============================================================================
        # LOSS FUNCTION: FocalMagnitudeExpandingLoss (FME-Loss)
        # ==============================================================================
        # Four-mechanism hybrid loss for asinh-transformed zero-inflated data:
        #   1. Asymmetric Residual Inflation  — counteracts asinh compression
        #   2. Charbonnier Core (gradient ∝ |r|^{p-1}) — never saturates
        #   3. Exponential Magnitude Weighting — inverts asinh compression
        #   4. Focal TN Suppression            — frees gradient budget from 85% zeros
        #
        # Stability: 99.9th-percentile per-sample clamp inside loss. Requires
        # gradient_clip_val ≥ 5.0 externally. batch_size ≥ 64 recommended.
        "loss_function": {"values": ["FocalMagnitudeExpandingLoss"]},
        "zero_threshold": {"values": [0.88]},
        # ── Charbonnier Core ──────────────────────────────────────────────
        # p (Lp exponent): Controls gradient growth in the tail.
        #   p=2.0 → MSE (linear gradient, least robust)
        #   p=1.5 → gradient ∝ √|r| (recommended sweet-spot)
        #   p=1.2 → gentler growth, closer to MAE
        # Must be strictly > 1.0.
        "p": {"values": [1.3, 1.5, 1.8]},
        # eps (smoothness): Width of the quadratic (MSE-like) zone near r=0.
        # Fixed at 0.1 (balanced). Rarely worth sweeping.
        "eps": {"values": [0.1]},
        # ── Residual Inflation ────────────────────────────────────────────
        # fn_inflation_power: Expands under-prediction residuals on large
        # targets before Charbonnier sees them.
        #   factor = 1 + (|y|/τ)^power
        #   0.0 = disabled (pure focal Charbonnier)
        #   0.5 = sqrt (gentle)
        #   0.7 = recommended
        #   1.0 = linear (aggressive, watch for instability)
        "fn_inflation_power": {
            "distribution": "uniform",
            "min": 0.3,
            "max": 1.0,
        },
        # ── Exponential Magnitude ─────────────────────────────────────────
        # magnitude_alpha: Exponential tilt strength.
        # w_mag = exp(α · |y| / τ). Inverts asinh compression.
        #   0.3 = mild  (10× at asinh=9)
        #   0.5 = moderate (166× at asinh=9, recommended)
        #   0.7 = aggressive (1600× at asinh=9)
        # Primary knob for underprediction correction.
        "magnitude_alpha": {
            "distribution": "uniform",
            "min": 0.3,
            "max": 0.7,
        },
        # ── Focal TN Suppression ──────────────────────────────────────────
        # focal_gamma: TN down-weighting. exp(-|r|)^γ.
        #   0.0 = no suppression (all zeros get full weight)
        #   2.0 = recommended
        #   3.0 = very aggressive
        "focal_gamma": {
            "distribution": "uniform",
            "min": 1.0,
            "max": 3.0,
        },
        # ── FN Residual Scaling ───────────────────────────────────────────
        # focal_gamma_fn: Scales FN weight by |r|^γ — larger misses get
        # progressively more penalty.
        #   0.0 = no scaling
        #   0.5 = sqrt (recommended)
        #   1.0 = linear
        "focal_gamma_fn": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 1.0,
        },
        # ── Class Weights ─────────────────────────────────────────────────
        # CRITICAL BALANCE: FN has three amplifiers (inflation × exp_mag ×
        # class weight) that can reach 80,000× effective weight. FP has
        # only class weight × FP focal (no inflation, no exp_mag since
        # target=0). If FP weight is too low relative to FN, the model
        # over-predicts conflict everywhere because FN spillover through
        # shared weights overwhelms FP pushback.
        #
        # Calibration: at mid-range params, a moderate conflict (asinh=5.3)
        # gets FN effective ≈ 18 × 20 = 360. To keep FP pushback within
        # ~1-2 orders of magnitude (letting the 85%/15% sample ratio help):
        #   FP_w × 54_zeros ≈ 360 × 10_conflicts × (spillover_fraction)
        #   FP_w ≈ 360 × 10 × 0.05 / 54 ≈ 3.3
        # So FP weight in [2, 8] keeps the balance reasonable.
        "non_zero_weight": {
            "distribution": "uniform",
            "min": 3.0,
            "max": 8.0,
        },
        # false_positive_weight: Must be high enough to counteract FN
        # spillover from shared network weights. FP focal modulation
        # inside the loss already softens small false alarms, so a high
        # base weight won't over-penalize near-threshold predictions.
        # Floor of 2.0 ensures meaningful pushback; ceiling of 8.0
        # prevents FP from dominating and causing underprediction.
        "false_positive_weight": {
            "distribution": "uniform",
            "min": 2.0,
            "max": 8.0,
        },
        # false_negative_weight: Additive on top of non_zero_weight.
        # This is the TERTIARY FN lever — inflation and magnitude_alpha
        # are the primary amplifiers. Keep moderate to avoid compounding
        # the already-extreme FN effective weight. Ceiling lowered from
        # 15 → 10 to reduce the FN/FP imbalance.
        "false_negative_weight": {
            "distribution": "uniform",
            "min": 1.0,
            "max": 10.0,
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