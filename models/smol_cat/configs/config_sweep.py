def get_sweep_config():
    """
    TiDE Hyperparameter Sweep Configuration - MagnitudeAwareCharbonnierLoss
    ========================================================================

    Strategy: Asinh-space regression with magnitude-aware asymmetric weighting
    ---------------------------------------------------------------------------

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
        "name": "smol_cat_tide_charbonnier_v18_cgm",
        "early_terminate": {"type": "hyperband", "min_iter": 30, "eta": 2},
        "metric": {"name": "time_series_wise_cgm_mean_sb", "goal": "minimize"},
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
        # Batch size interacts with loss + rare signal: larger batches give
        # more stable gradient estimates (critical when non-zero samples are
        # <5% of each batch). Smaller batches give noisier gradients that
        # can help escape flat minima.
        "batch_size": {"values": [32, 64, 128]},
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
            "max": 8e-4,
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
        # Charbonnier gradients grow as |r|^{p-1} (never cap). Clipping
        # at 1.0 can re-introduce the hard gradient ceiling we designed
        # the loss to avoid. Sweep to find the tradeoff between stability
        # and preserving the Charbonnier advantage.
        "gradient_clip_val": {"values": [1.0, 5.0, 10.0]},
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
        # LOSS FUNCTION: MagnitudeAwareCharbonnierLoss (Smooth Lp)
        # ==============================================================================
        # Replaces MagnitudeAwareHuberLoss. Key advantage: gradient never
        # saturates (∝ |r|^{p-1}), eliminating the spike suppression that
        # Huber/LogCosh cause during extended training. See Barron (2019)
        # "A General and Adaptive Robust Loss Function", CVPR.
        #
        # Core: L(r) = [(r² + ε²)^{p/2} − εᵖ] / [(p/2)·ε^{p−2}]
        # Normalised so L(r) ≈ r² near origin regardless of (p, ε).
        "loss_function": {"values": ["MagnitudeAwareCharbonnierLoss"]},
        "zero_threshold": {"values": [0.88]},
        # p (Lp exponent): Controls gradient growth in the tail.
        #   p=2.0 → MSE (linear gradient, least robust)
        #   p=1.5 → gradient ∝ √|r| (recommended sweet-spot)
        #   p→1.0 → approaches L1 (gradient saturates — defeats purpose)
        # Must be strictly > 1.0 to ensure non-saturating gradients.
        "p": {"values": [1.2, 1.5, 1.7, 2.0]},
        # eps (smoothness): Width of the quadratic (MSE-like) zone.
        #   Small ε → narrow quadratic zone, aggressive sub-linear onset
        #   Large ε → wide quadratic zone, MSE-like over typical residuals
        # For asinh-transformed data (residuals typically 0–7):
        #   0.01 = very aggressive, 0.1 = balanced, 0.5 = conservative
        "eps": {"values": [0.01, 0.1, 0.5]},
        "non_zero_weight": {
            "distribution": "uniform",
            "min": 2.0,
            "max": 10.0,
        },
        # false_positive_weight: Raised floor to 0.6.
        # Low FP weight allows unconstrained over-prediction, which ironically
        # contributes to flatline: the model learns it's "free" to predict
        # low values without penalty, so near-zero is always safe.
        # Higher FP weight forces the model to commit — if it predicts
        # conflict, it should be right, but when it does predict, the
        # predictions carry more signal.
        "false_positive_weight": {
            "distribution": "uniform",
            "min": 0.6,
            "max": 1.0,
        },
        # false_negative_weight: With Charbonnier the gradient never caps,
        # so FN weight can be lower than with Huber while still breaking
        # through. But we keep a meaningful floor to ensure the model
        # prioritises rare conflict events over the zero-majority.
        "false_negative_weight": {
            "distribution": "uniform",
            "min": 2.0,
            "max": 10.0,
        },
        # magnitude_exponent: Power-law importance weighting.
        # With Charbonnier, this no longer compensates for gradient
        # saturation — purely controls large-vs-small event weighting.
        # Widened to include 0.0 (no magnitude scaling) as a baseline:
        # need to verify magnitude weighting helps with a non-saturating core.
        "magnitude_exponent": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 1.0,
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