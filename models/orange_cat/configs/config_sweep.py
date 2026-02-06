def get_sweep_config():
    """
    TiDE (Time-series Dense Encoder) Hyperparameter Sweep Configuration
    =====================================================================
    Optimized for country-month level rare conflict fatality forecasting.

    Problem Characteristics:
    ------------------------
    - ~200 time series (countries)
    - Zero-inflated target: Most country-months have zero fatalities
    - Heavy right skew: When conflicts occur, fatality counts vary enormously
    - Rare signal: Model must learn maximally from scarce non-zero events
    - 36-month forecast horizon with monthly resolution

    TiDE Architecture Overview:
    ---------------------------
    TiDE is a lightweight MLP-based model designed for long-horizon forecasting:

    1. DENSE ENCODER:
       - Projects historical time steps into dense representations
       - Separate temporal processing for past and future contexts
       - Avoids attention overhead while maintaining quality

    2. RESIDUAL CONNECTIONS (Skip Connections):
       - Direct path from input to output
       - Ensures gradient flow even if main network struggles
       - Critical insight: Skip connections will always learn;
         the challenge is making the main network also contribute

    3. TEMPORAL PROCESSING:
       - temporal_width controls local receptive field
       - temporal_hidden_size controls representation capacity
       - Separate processing for past context and future horizon

    4. DECODER:
       - Projects encoded representations to forecasts
       - Relatively simple compared to encoder

    Key Design Decisions:
    ---------------------
    1. SCALING: AsinhTransform->MinMaxScaler for target
       - Asinh handles zeros naturally (unlike log)
       - Asinh(x) ≈ x for small x, ≈ ln(2x) for large x
       - MinMax bounds output to [0,1] for stable gradients
       - 1 fatality → ~0.11 after transform (calibrates zero_threshold)

    2. LOSS: WeightedPenaltyHuberLoss with high delta (0.8-1.0)
       - Full L2 behavior maximizes gradient signal from rare spikes
       - Asymmetric weights: Missing conflict (FN) penalized more than false alarm (FP)
       - Combined weights: TN=1x, FP=0.5-1x, TP=4-7x, FN=8-56x

    3. REGULARIZATION: Minimal (weight_decay=0, low dropout)
       - Scarce signal means we cannot afford to suppress rare pattern neurons
       - Previous experiments showed weight collapse with weight_decay > 0
       - Layer norm optional: can help or hurt depending on data

    4. ARCHITECTURE: Moderate sizes, controlled depth
       - ~200 series cannot support very large hidden sizes
       - Shallow encoder/decoder (1-3 layers) to prevent overfitting
       - Temporal widths of 4-8 capture local patterns

    Weight Collapse Prevention:
    ---------------------------
    Previous experiments showed weights shrinking to ~1e-34 due to:
    - Excessive weight_decay (was 5e-4, now 0)
    - Combined regularization (dropout + weight_decay + layer_norm)
    - MinMaxScaler compressing gradients

    Fixes applied:
    - weight_decay = 0
    - Low dropout (0.05-0.15)
    - Layer norm optional (explore both)
    - Higher delta for stronger gradients

    Hyperband Early Termination:
    ----------------------------
    - min_iter=20: Gives TiDE time to find patterns in scarce signal
    - eta=2: Moderately aggressive pruning (keeps top 50% each round)

    Search Space Size: ~186,000 discrete combinations + continuous parameters
    Estimated sweep runs needed: 400-600 for good Bayesian coverage

    Returns:
        sweep_config (dict): WandB sweep configuration dictionary
    """

    sweep_config = {
        "method": "bayes",
        "name": "orange_cat_tide_v6_mtd",
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 20,
            "eta": 2,
        },
        "metric": {"name": "time_series_wise_mtd_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        # steps: 36-month forecast horizon (standard for conflict forecasting)
        "steps": {"values": [[*range(1, 36 + 1)]]},
        # input_chunk_length: Historical context window
        # - TiDE uses dense encoding of the full input sequence
        # - 36 months (3 years): Captures annual cycles, recent trends
        # - 48 months (4 years): Captures electoral cycles, medium-term patterns
        # - TiDE is efficient with moderate sequence lengths
        "input_chunk_length": {"values": [36, 48]},
        "output_chunk_shift": {"values": [0]},  # No gap between input and forecast
        "mc_dropout": {"values": [True]},  # Monte Carlo dropout for uncertainty
        "random_state": {"values": [67]},  # Reproducibility
        # ==============================================================================
        # TRAINING BASICS
        # ==============================================================================
        # batch_size: Samples per gradient update
        # - Larger batches help zero-inflated data see more non-zero events
        # - TiDE is memory-efficient (no attention matrices)
        # - Range 512-4096 provides good coverage
        "batch_size": {"values": [1024, 2048, 4096]},
        # n_epochs: Maximum training epochs
        # - 150 epochs provides headroom; early stopping triggers before max
        # - Increased from 100 since we reduced regularization
        "n_epochs": {"values": [150]},
        # early_stopping_patience: Epochs without improvement before stopping
        # - Higher patience (15-25) for scarce signal
        # - Rare conflict patterns may take time to emerge in validation
        "early_stopping_patience": {"values": [20]},
        # early_stopping_min_delta: Minimum improvement to count as progress
        # - Small values (5e-5 to 1e-4) appropriate for [0,1] scaled loss
        "early_stopping_min_delta": {"values": [0.0001]},
        "force_reset": {"values": [True]},  # Clean model state each sweep run
        # ==============================================================================
        # OPTIMIZER / LEARNING RATE SCHEDULE
        # ==============================================================================
        # lr: Learning rate (log-uniform distribution for proper exploration)
        # - 5e-5: Conservative, stable learning
        # - 1e-3: Aggressive upper bound
        # - TiDE typically works well in 1e-4 to 5e-4 range
        "lr": {
            "distribution": "log_uniform_values",
            "min": 5e-5,
            "max": 1e-3,
        },
        # weight_decay: L2 regularization on weights
        # DISABLED (set to 0) because:
        # - CRITICAL: High weight_decay caused weight collapse (weights → 1e-34)
        # - Scarce signal means neurons learning rare patterns are precious
        # - Combined with dropout + layer_norm, weight_decay was too much
        # - Rule of thumb violated: weight_decay was 100x+ larger than lr
        "weight_decay": {"values": [0]},
        # lr_scheduler: ReduceLROnPlateau configuration
        # - factor=0.5: Halve LR when stuck (standard, well-tested)
        # - patience=8: Wait 8 epochs before reducing
        # - min_lr=1e-6: Floor prevents LR from becoming negligible
        "lr_scheduler_factor": {"values": [0.5]},
        "lr_scheduler_patience": {"values": [8]},
        "lr_scheduler_min_lr": {"values": [1e-6]},
        # gradient_clip_val: Maximum gradient norm
        # - Prevents exploding gradients
        # - TiDE has stable gradients due to residual connections
        # - Range 0.5-1.5 is conservative for [0,1] scaled data
        "gradient_clip_val": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1.5,
        },
        # ==============================================================================
        # FEATURE SCALING
        # ==============================================================================
        # feature_scaler: Global default (None = use feature_scaler_map)
        "feature_scaler": {"values": [None]},
        # target_scaler: AsinhTransform->MinMaxScaler
        # - AsinhTransform: asinh(x) = ln(x + sqrt(x² + 1))
        #   * Handles zeros naturally (asinh(0) = 0)
        #   * Linear near zero, logarithmic for large values
        # - MinMaxScaler: Bounds to [0,1] after Asinh
        #   * Stabilizes neural network training
        #   * Makes loss values interpretable
        #   * Calibration: asinh(1)≈0.88 → after MinMax ≈0.11
        "target_scaler": {"values": ["AsinhTransform->MinMaxScaler"]},
        # feature_scaler_map: Per-feature scaling based on distribution characteristics
        "feature_scaler_map": {
            "values": [
                {
                    # AsinhTransform->MinMaxScaler: For zero-inflated and right-skewed features
                    # These have many zeros and occasional extreme values
                    "AsinhTransform->MinMaxScaler": [
                        # Conflict fatality counts (zero-inflated, extreme outliers)
                        "lr_ged_sb",
                        "lr_ged_ns",
                        "lr_ged_os",
                        "lr_acled_sb",
                        "lr_acled_sb_count",
                        "lr_acled_os",
                        "lr_ged_sb_tsum_24",  # 24-month cumulative
                        "lr_splag_1_ged_sb",
                        "lr_splag_1_ged_os",
                        "lr_splag_1_ged_ns",  # Spatial lags
                        # Economic indicators with extreme skew (GDP spans orders of magnitude)
                        "lr_wdi_ny_gdp_mktp_kd",
                        "lr_wdi_nv_agr_totl_kn",
                        "lr_wdi_sm_pop_netm",
                        "lr_wdi_sm_pop_refg_or",
                        # Mortality rates (positive, skewed)
                        "lr_wdi_sp_dyn_imrt_fe_in",
                        # Token counts from text analysis (zero-inflated)
                        "lr_topic_tokens_t1",
                        "lr_topic_tokens_t1_splag",
                    ],
                    # MinMaxScaler: For bounded or roughly symmetric features
                    # These are already in reasonable ranges, just need [0,1] normalization
                    "MinMaxScaler": [
                        # WDI percentages (0-100 scale)
                        "lr_wdi_sl_tlf_totl_fe_zs",
                        "lr_wdi_se_enr_prim_fm_zs",
                        "lr_wdi_sp_urb_totl_in_zs",
                        "lr_wdi_sh_sta_maln_zs",
                        "lr_wdi_sh_sta_stnt_zs",
                        "lr_wdi_dt_oda_odat_pc_zs",
                        "lr_wdi_ms_mil_xpnd_gd_zs",
                        # V-Dem indices (already 0-1 normalized)
                        "lr_vdem_v2x_horacc",
                        "lr_vdem_v2xnp_client",
                        "lr_vdem_v2x_veracc",
                        "lr_vdem_v2x_divparctrl",
                        "lr_vdem_v2xpe_exlpol",
                        "lr_vdem_v2x_diagacc",
                        "lr_vdem_v2xpe_exlgeo",
                        "lr_vdem_v2xpe_exlgender",
                        "lr_vdem_v2xpe_exlsocgr",
                        "lr_vdem_v2x_ex_party",
                        "lr_vdem_v2x_genpp",
                        "lr_vdem_v2xeg_eqdr",
                        "lr_vdem_v2xcl_prpty",
                        "lr_vdem_v2xeg_eqprotec",
                        "lr_vdem_v2x_ex_military",
                        "lr_vdem_v2xcl_dmove",
                        "lr_vdem_v2x_clphy",
                        "lr_vdem_v2x_hosabort",
                        "lr_vdem_v2xnp_regcorr",
                        # Topic model theta values (probability distributions, sum to 1)
                        "lr_topic_ste_theta0",
                        "lr_topic_ste_theta1",
                        "lr_topic_ste_theta2",
                        "lr_topic_ste_theta3",
                        "lr_topic_ste_theta4",
                        "lr_topic_ste_theta5",
                        "lr_topic_ste_theta6",
                        "lr_topic_ste_theta7",
                        "lr_topic_ste_theta8",
                        "lr_topic_ste_theta9",
                        "lr_topic_ste_theta10",
                        "lr_topic_ste_theta11",
                        "lr_topic_ste_theta12",
                        "lr_topic_ste_theta13",
                        "lr_topic_ste_theta14",
                        "lr_topic_ste_theta0_stock_t1_splag",
                        "lr_topic_ste_theta1_stock_t1_splag",
                        "lr_topic_ste_theta2_stock_t1_splag",
                        "lr_topic_ste_theta3_stock_t1_splag",
                        "lr_topic_ste_theta4_stock_t1_splag",
                        "lr_topic_ste_theta5_stock_t1_splag",
                        "lr_topic_ste_theta6_stock_t1_splag",
                        "lr_topic_ste_theta7_stock_t1_splag",
                        "lr_topic_ste_theta8_stock_t1_splag",
                        "lr_topic_ste_theta9_stock_t1_splag",
                        "lr_topic_ste_theta10_stock_t1_splag",
                        "lr_topic_ste_theta11_stock_t1_splag",
                        "lr_topic_ste_theta12_stock_t1_splag",
                        "lr_topic_ste_theta13_stock_t1_splag",
                        "lr_topic_ste_theta14_stock_t1_splag",
                        # Population growth rate (small range, can be negative)
                        "lr_wdi_sp_pop_grow",
                    ],
                }
            ]
        },
        # ==============================================================================
        # TiDE ENCODER ARCHITECTURE
        # ==============================================================================
        # num_encoder_layers: Depth of the dense encoder
        # - Encoder projects input sequence into dense representations
        # - 1 layer: Simple, fast, less overfitting
        # - 2 layers: Moderate capacity for pattern extraction
        # - 3 layers: Deeper features (may overfit with scarce signal)
        "num_encoder_layers": {"values": [1, 2, 3]},
        # num_decoder_layers: Depth of the decoder
        # - Decoder projects encoded representations to forecasts
        # - Generally shallower than encoder (forecasting is "simpler")
        # - 1-2 layers sufficient for most time series tasks
        "num_decoder_layers": {"values": [1, 2]},
        # decoder_output_dim: Output dimension of decoder before final projection
        # - Controls capacity of forecast generation
        # - 16: Lightweight, fast
        # - 32: Balanced
        # - 64: Higher capacity for complex forecast patterns
        "decoder_output_dim": {"values": [16, 32, 64]},
        # hidden_size: Hidden dimension of encoder MLP layers
        # - Controls main encoder capacity
        # - 16: Very lightweight
        # - 32-64: Balanced for ~200 series
        # - 128: Higher capacity (monitor for overfitting)
        # - Avoid larger sizes to prevent overfitting with scarce signal
        "hidden_size": {"values": [32, 64, 128]},
        # ==============================================================================
        # TiDE TEMPORAL PROCESSING
        # ==============================================================================
        # temporal_width_past: Local receptive field for past context
        # - How many adjacent past time steps to consider together
        # - 4: ~1 quarter of local context
        # - 6: ~half year of local context
        # - 8: ~2/3 year of local context
        "temporal_width_past": {"values": [4, 6, 8]},
        # temporal_width_future: Local receptive field for future horizon
        # - How many adjacent future time steps to consider together
        # - Similar reasoning to temporal_width_past
        "temporal_width_future": {"values": [4, 6, 8]},
        # temporal_hidden_size_past: Hidden dim for past temporal processing
        # - Capacity for learning patterns in historical context
        # - 32: Conservative
        # - 64: Balanced
        # - 128: Higher capacity for complex temporal patterns
        "temporal_hidden_size_past": {"values": [32, 64, 128]},
        # temporal_hidden_size_future: Hidden dim for future temporal processing
        # - Capacity for learning patterns in forecast horizon
        # - Generally similar or slightly smaller than past
        "temporal_hidden_size_future": {"values": [32, 64, 128]},
        # temporal_decoder_hidden: Hidden dim of temporal decoder
        # - Final temporal processing before output
        # - 64-256 range covers lightweight to high capacity
        "temporal_decoder_hidden": {"values": [64, 128, 256]},
        # ==============================================================================
        # TiDE REGULARIZATION
        # ==============================================================================
        # use_layer_norm: Whether to apply layer normalization
        # - True: Stabilizes training, helps with varying input scales
        # - False: Simpler, may prevent over-normalization
        # - Worth exploring both for zero-inflated data
        "use_layer_norm": {"values": [True, False]},
        # dropout: Dropout rate throughout TiDE
        # - LOW values (0.05-0.15) for scarce signal
        # - High dropout would suppress neurons learning rare conflict patterns
        # - Combined with weight_decay=0, this is the main regularization
        "dropout": {"values": [0.05, 0.15]},
        # use_static_covariates: Whether to use static (time-invariant) features
        # - True: Leverages country-level constants (geography, etc.)
        # - False: Simpler model, may generalize better if static features noisy
        "use_static_covariates": {"values": [False, True]},
        # use_reversible_instance_norm: Per-instance normalization
        # - Normalizes each time series independently before processing
        # - "Reversible" stores stats to invert normalization on output
        # - True: Helps with non-stationary data (conflict patterns evolve)
        # - False: Simpler, may generalize better if series are comparable
        "use_reversible_instance_norm": {"values": [True, False]},
        # ==============================================================================
        # LOSS FUNCTION: WeightedPenaltyHuberLoss
        # ==============================================================================
        # Custom loss designed for zero-inflated rare event forecasting.
        # Combines Huber loss with asymmetric weighting based on prediction type.
        #
        # Weight multiplication logic:
        # - True Negative (zero→zero): weight = 1.0 (baseline)
        # - False Positive (zero→non-zero): weight = false_positive_weight (0.5-1.0)
        # - True Positive (non-zero→non-zero): weight = non_zero_weight (4-7)
        # - False Negative (non-zero→zero): weight = non_zero_weight × fn_weight (8-56)
        #
        # This encourages the model to:
        # 1. Learn strongly from actual conflict events (high non_zero_weight)
        # 2. Heavily penalize missing conflicts (high FN penalty)
        # 3. Be somewhat forgiving of false alarms (low FP weight encourages exploration)
        "loss_function": {"values": ["TweedieLoss"]},
        # zero_threshold: Scaled value below which predictions count as "zero"
        # - After AsinhTransform->MinMaxScaler, 1 fatality ≈ 0.11
        # - Range 0.08-0.23 spans 0-5 fatalities threshold and allows some margin for uncertainty
        # - Lower threshold = stricter zero classification
        "zero_threshold": {
            "distribution": "uniform",
            "min": 0.08,  # 0 fatalities threshold after scaling
            "max": 0.23,  # 5 fatalities threshold after scaling
        },
        # delta: Huber loss transition point (L2 inside delta, L1 outside)
        # - Range 0.8-1.0 gives nearly pure L2 behavior for [0,1] scaled data
        # - Full L2 maximizes gradient signal from every error
        # - Important for learning from rare spikes where every gradient counts
        # "delta": {
        #     "distribution": "uniform",
        #     "min": 0.8,
        #     "max": 1.0,
        # },
        # non_zero_weight: Multiplier for non-zero actual values
        # - Fixed at 5.0 to reduce search dimensions
        # - Conflicts contribute 5x more to loss than zeros (counteracts class imbalance)
        # - FP and FN weights are tuned relative to this baseline
        # "non_zero_weight": {"values": [5.0]},
        "non_zero_weight": {
            "distribution": "log_uniform_values",
            "min": 1.0,
            "max": 5.0,
        },
        # false_positive_weight: Multiplier when predicting non-zero for actual zero
        # - Range 0.5-1.0 (at or below baseline)
        # - Values <1.0 encourage model to "explore" non-zero predictions
        # - Helps escape local minimum of predicting all zeros
        # "false_positive_weight": {
        #     "distribution": "uniform",
        #     "min": 0.5,
        #     "max": 1.0,
        # },
        "false_positive_weight": {"values": [1.0]},
        # false_negative_weight: Additional multiplier for missing actual conflicts
        # - Applied ON TOP of non_zero_weight: total FN penalty = non_zero × fn_weight
        # - Range 2-8 gives total FN weight of 8-56x baseline
        # - Highest penalty because missing conflicts is operationally costly
        # "false_negative_weight": {
        #     "distribution": "uniform",
        #     "min": 2.0,
        #     "max": 8.0,
        # },
        "false_negative_weight": {"values": [1.0]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config
