def get_sweep_config():
    """
    TiDE Hyperparameter Sweep Configuration - MagnitudeAwareHuberLoss
    ==================================================================

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
        "name": "cool_cat_tide_mahub_v17_msle",
        "early_terminate": {"type": "hyperband", "min_iter": 30, "eta": 2},
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        "input_chunk_length": {"values": [36]},
        "output_chunk_shift": {"values": [0]},
        "random_state": {"values": [67]},
        "output_chunk_length": {"values": [36]},
        "optimizer_cls": {"values": ["Adam"]},
        # mc_dropout=False: Deterministic inference. Dropout at inference
        # suppresses activations by (1-p), compounding signal loss across
        # decoder layers. With the flatline problem, every bit of signal matters.
        "mc_dropout": {"values": [False]},
        "num_samples": {"values": [1]},
        "n_jobs": {"values": [-1]},
        # ==============================================================================
        # TRAINING
        # ==============================================================================
        "batch_size": {"values": [64]},
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [40]},
        "early_stopping_min_delta": {"values": [0.0001]},
        "force_reset": {"values": [True]},
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        "lr": {
            "distribution": "log_uniform_values",
            "min": 8e-5,
            "max": 4e-4,
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
        "gradient_clip_val": {"values": [1.0]},
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
        "temporal_width_future": {"values": [24, 48, 64]},
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
        "dropout": {
            "distribution": "uniform",
            "min": 0.15,
            "max": 0.28,
        },
        "use_static_covariates": {"values": [True]},
        # Without autoregressive chaining, RevIN is the primary mechanism to
        # prevent collapse to the global unconditional mean.
        #
        # How it helps: RevIN normalizes each series to zero-mean/unit-var at
        # input, then REVERSES the normalization at output. So even if the
        # decoder predicts a flat "0.3" for all steps in normalized space,
        # the denormalization rescales it to Sudan's range (~500+) and
        # Norway's range (~0). The decoder only needs to learn RELATIVE
        # temporal patterns, not absolute magnitudes.
        #
        # Risk: Zeros become negative in normalized space, distorting the
        # zero_threshold boundary. But the flatline is worse than threshold
        # distortion — a threshold can be recalibrated, a flatline cannot.
        #
        # Sweep both to let Bayes measure the tradeoff.
        "use_reversible_instance_norm": {"values": [True, False]},
        # ==============================================================================
        # LOSS FUNCTION: MagnitudeAwareHuberLoss
        # ==============================================================================
        "loss_function": {"values": ["MagnitudeAwareHuberLoss"]},
        "zero_threshold": {"values": [0.88]},
        "delta": {
            "distribution": "uniform",
            "min": 1.0,
            "max": 2.0,
        },
        "non_zero_weight": {
            "distribution": "uniform",
            "min": 3.0,
            "max": 6.0,
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
        # false_negative_weight: Raised floor to 2.0.
        # This is the most direct anti-flatline lever in the loss function.
        # Higher FN weight = steeper gradient when the model predicts zero
        # for active conflict. Forces the decoder's distant-horizon outputs
        # to maintain non-zero predictions for conflict-active series.
        "false_negative_weight": {
            "distribution": "uniform",
            "min": 2.0,
            "max": 10.0,
        },
        # magnitude_exponent: Shifted upward.
        # Higher exponent = more gradient for large events = decoder is
        # punished harder for flatline on high-casualty countries like Sudan.
        # At exp=0.8, a 1000-fatality miss gets 8.2x the gradient of a
        # 1-fatality miss (vs 4.2x at exp=0.5).
        "magnitude_exponent": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 0.85,
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