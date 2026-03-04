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
        # Batch size: Fixed at 64. FME-Loss requires ≥64 for stable
        # gradients with three unbounded multipliers.
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
        # FME-Loss has three unbounded multipliers. Fixed at 5.0 — the
        # safe floor. 10.0 is too loose for the worst-case alpha×inflation.
        "gradient_clip_val": {"values": [5.0]},
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
        # Layer norm: Fixed at True. Stabilizes Charbonnier gradient flow.
        "use_layer_norm": {"values": [True]},
        # Dropout: SWEPT but narrowed. High dropout kills rare conflict
        # signal. Light regularization only.
        "dropout": {
            "distribution": "uniform",
            "min": 0.05,
            "max": 0.15,
        },
        "use_static_covariates": {"values": [True]},
        # RevIN disabled: Per-series normalization distorts the zero_threshold 
        # in Asinh space. The FME-Loss handles magnitude scaling natively, making 
        # RevIN redundant and potentially harmful to the loss function's logic.
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
        #
        # FIXED loss params (well-understood or secondary):
        #
        # p=1.5: Theoretical sweet spot for Charbonnier (sqrt gradient growth).
        "p": {"values": [1.5]},
        "eps": {"values": [0.1]},
        # focal_gamma=2.0: Lin et al. ICCV 2017 default, well-validated.
        "focal_gamma": {"values": [2.0]},
        # focal_gamma_fn=0.5: sqrt scaling, balanced.
        "focal_gamma_fn": {"values": [0.5]},
        # Class weights: Fixed at calibrated midpoints. The exponential
        # magnitude + inflation are the primary FN amplifiers. Class weights
        # are tertiary — fixing them removes 2 dimensions.
        "non_zero_weight": {"values": [5.0]},
        "false_negative_weight": {"values": [5.0]},
        #
        # SWEPT loss params (the 3 primary levers):
        #
        # ── magnitude_alpha ───────────────────────
        # THE most important knob. Exponential tilt strength.
        # w_mag = exp(α · |y| / τ). Inverts asinh compression.
        #   0.3 = mild  (10× at asinh=9)
        #   0.5 = moderate (166× at asinh=9, recommended)
        #   0.7 = aggressive (1600× at asinh=9)
        "magnitude_alpha": {
            "distribution": "uniform",
            "min": 0.3,
            "max": 0.7,
        },
        # ── fn_inflation_power  ────────────────────
        # Expands under-prediction residuals before Charbonnier sees them.
        #   factor = 1 + (|y|/τ)^power
        #   0.3 = gentle, 0.7 = recommended, 1.0 = aggressive
        "fn_inflation_power": {
            "distribution": "uniform",
            "min": 0.3,
            "max": 1.0,
        },
        # ── false_positive_weight ───────────────────
        # Must scale with the FN amplifiers (alpha × inflation) to prevent
        # "Global Escalation." FP focal modulation inside the loss already
        # softens small false alarms, so a high base weight won't
        # over-penalize near-threshold predictions.
        #   At alpha=0.7, inflation=1.0: effective FN ≈ 10,000×
        #   FP weight of 3.0 would be too loose; 8.0 may over-suppress.
        # Let Bayes find the equilibrium.
        "false_positive_weight": {
            "distribution": "uniform",
            "min": 3.0,
            "max": 8.0,
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