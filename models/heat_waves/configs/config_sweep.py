def get_sweep_config():
    """
    TFT (Temporal Fusion Transformer) Hyperparameter Sweep Configuration
    =====================================================================

    Problem Characteristics:
    ------------------------
    - ~200 time series (countries)
    - Zero-inflated target: Most country-months have zero fatalities
    - Heavy right skew: When conflicts occur, fatality counts vary enormously
    - Rare signal: Model must learn maximally from scarce non-zero events
    - 36-month forecast horizon with monthly resolution

    TFT Architecture Strengths:
    ---------------------------
    - Variable Selection Networks: Automatically learns feature importance
    - Interpretable Multi-Head Attention: Shows which time steps matter
    - Gated Residual Networks: Controls information flow, prevents gradient issues
    - Static/Dynamic covariate handling: Separates country-level vs time-varying features
    - Quantile outputs: Native uncertainty quantification

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
       - Gradient clipping prevents instability without killing learning

    4. ARCHITECTURE: Moderate size, shallow depth
       - ~200 series cannot support very large models (overfitting risk)
       - TFT's attention does heavy lifting; LSTM backbone can be shallow
       - hidden_size/nhead >= 16 required for stable attention

    Hyperband Early Termination:
    ----------------------------
    - min_iter=20: Gives models time to find patterns in scarce signal
    - eta=2: Moderately aggressive pruning (keeps top 50% each round)

    Search Space Size: ~62,000 discrete combinations + continuous parameters
    Estimated sweep runs needed: 200-400 for good Bayesian coverage

    Returns:
        sweep_config (dict): WandB sweep configuration dictionary
    """

    sweep_config = {
        "method": "bayes",
        "name": "heat_waves_tft_v5_mtd",
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
        # - 36 months (3 years): Captures annual cycles, recent escalation
        # - 48 months (4 years): Captures electoral cycles, medium-term trends
        # - 60 months (5 years): Captures longer political/economic cycles
        # TFT's attention mechanism benefits from longer context to learn
        # which historical periods are predictive of future conflict
        "input_chunk_length": {"values": [36, 48, 60]},

        "output_chunk_shift": {"values": [0]},  # No gap between input and forecast
        "output_chunk_length": {"values": [36]},  # Must match steps
        "random_state": {"values": [42]},  # Reproducibility
        "mc_dropout": {"values": [True]},  # Enable Monte Carlo dropout for uncertainty

        # ==============================================================================
        # TRAINING BASICS
        # ==============================================================================
        # batch_size: Larger batches help zero-inflated data
        # - More samples per batch = higher chance of seeing non-zero events
        # - Stabilizes gradient estimates when most samples are zeros
        # - Range 256-2048 balances GPU memory with gradient quality
        "batch_size": {"values": [256, 512, 1024, 2048]},

        # n_epochs: Maximum training epochs (early stopping will trigger before)
        # TFT converges slower than simpler models due to attention complexity
        "n_epochs": {"values": [150]},

        # early_stopping_patience: Epochs without improvement before stopping
        # - Higher patience (15-25) for scarce signal: model may plateau then improve
        # - Rare conflict patterns may take time to emerge in validation metrics
        "early_stopping_patience": {"values": [20]},

        # early_stopping_min_delta: Minimum improvement to count as progress
        # - Small values (5e-5 to 1e-4) appropriate for [0,1] scaled loss
        # - Too large = premature stopping; too small = wasted compute
        "early_stopping_min_delta": {"values": [0.0001]},

        "force_reset": {"values": [True]},  # Clean model state each sweep run

        # ==============================================================================
        # OPTIMIZER / LEARNING RATE SCHEDULE
        # ==============================================================================
        # lr: Learning rate (log-uniform distribution for proper exploration)
        # - 5e-5: Conservative, stable but slow learning
        # - 1e-3: Aggressive, faster but risk of instability
        # TFT typically works well in 1e-4 to 5e-4 range
        "lr": {
            "distribution": "log_uniform_values",
            "min": 5e-5,
            "max": 1e-3,
        },

        # weight_decay: L2 regularization on weights
        # DISABLED (set to 0) because:
        # - Scarce signal means neurons learning rare patterns are precious
        # - Weight decay penalizes large weights that may encode important patterns
        # - Previous experiments showed weight collapse with weight_decay > 0
        "weight_decay": {"values": [0]},

        # lr_scheduler: ReduceLROnPlateau configuration
        # - factor=0.5: Halve LR when stuck (standard, well-tested)
        # - patience=8: Wait 8 epochs before reducing (allows temporary plateaus)
        # - min_lr=1e-6: Floor prevents LR from becoming negligible
        "lr_scheduler_factor": {"values": [0.5]},
        "lr_scheduler_patience": {"values": [8]},
        "lr_scheduler_min_lr": {"values": [1e-6]},

        # gradient_clip_val: Maximum gradient norm
        # - Prevents exploding gradients (especially in attention layers)
        # - Range 0.5-1.5 is conservative; TFT has stable gradients
        # - Lower values = more stable but potentially slower learning
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
        #   * Symmetric: works for negative values if needed
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
                        "lr_ged_sb", "lr_ged_ns", "lr_ged_os",
                        "lr_acled_sb", "lr_acled_sb_count", "lr_acled_os",
                        "lr_ged_sb_tsum_24",  # 24-month cumulative
                        "lr_splag_1_ged_sb", "lr_splag_1_ged_os", "lr_splag_1_ged_ns",  # Spatial lags
                        # Economic indicators with extreme skew (GDP spans orders of magnitude)
                        "lr_wdi_ny_gdp_mktp_kd", "lr_wdi_nv_agr_totl_kn",
                        "lr_wdi_sm_pop_netm", "lr_wdi_sm_pop_refg_or",
                        # Mortality rates (positive, skewed)
                        "lr_wdi_sp_dyn_imrt_fe_in",
                        # Token counts from text analysis (zero-inflated)
                        "lr_topic_tokens_t1", "lr_topic_tokens_t1_splag",
                    ],
                    # MinMaxScaler: For bounded or roughly symmetric features
                    # These are already in reasonable ranges, just need [0,1] normalization
                    "MinMaxScaler": [
                        # WDI percentages (0-100 scale)
                        "lr_wdi_sl_tlf_totl_fe_zs", "lr_wdi_se_enr_prim_fm_zs",
                        "lr_wdi_sp_urb_totl_in_zs", "lr_wdi_sh_sta_maln_zs",
                        "lr_wdi_sh_sta_stnt_zs", "lr_wdi_dt_oda_odat_pc_zs",
                        "lr_wdi_ms_mil_xpnd_gd_zs",
                        # V-Dem indices (already 0-1 normalized)
                        "lr_vdem_v2x_horacc", "lr_vdem_v2xnp_client", "lr_vdem_v2x_veracc",
                        "lr_vdem_v2x_divparctrl", "lr_vdem_v2xpe_exlpol", "lr_vdem_v2x_diagacc",
                        "lr_vdem_v2xpe_exlgeo", "lr_vdem_v2xpe_exlgender", "lr_vdem_v2xpe_exlsocgr",
                        "lr_vdem_v2x_ex_party", "lr_vdem_v2x_genpp", "lr_vdem_v2xeg_eqdr",
                        "lr_vdem_v2xcl_prpty", "lr_vdem_v2xeg_eqprotec", "lr_vdem_v2x_ex_military",
                        "lr_vdem_v2xcl_dmove", "lr_vdem_v2x_clphy", "lr_vdem_v2x_hosabort",
                        "lr_vdem_v2xnp_regcorr",
                        # Topic model theta values (probability distributions, sum to 1)
                        "lr_topic_ste_theta0", "lr_topic_ste_theta1", "lr_topic_ste_theta2",
                        "lr_topic_ste_theta3", "lr_topic_ste_theta4", "lr_topic_ste_theta5",
                        "lr_topic_ste_theta6", "lr_topic_ste_theta7", "lr_topic_ste_theta8",
                        "lr_topic_ste_theta9", "lr_topic_ste_theta10", "lr_topic_ste_theta11",
                        "lr_topic_ste_theta12", "lr_topic_ste_theta13", "lr_topic_ste_theta14",
                        "lr_topic_ste_theta0_stock_t1_splag", "lr_topic_ste_theta1_stock_t1_splag",
                        "lr_topic_ste_theta2_stock_t1_splag", "lr_topic_ste_theta3_stock_t1_splag",
                        "lr_topic_ste_theta4_stock_t1_splag", "lr_topic_ste_theta5_stock_t1_splag",
                        "lr_topic_ste_theta6_stock_t1_splag", "lr_topic_ste_theta7_stock_t1_splag",
                        "lr_topic_ste_theta8_stock_t1_splag", "lr_topic_ste_theta9_stock_t1_splag",
                        "lr_topic_ste_theta10_stock_t1_splag", "lr_topic_ste_theta11_stock_t1_splag",
                        "lr_topic_ste_theta12_stock_t1_splag", "lr_topic_ste_theta13_stock_t1_splag",
                        "lr_topic_ste_theta14_stock_t1_splag",
                        # Population growth rate (small range, can be negative)
                        "lr_wdi_sp_pop_grow",
                    ],
                }
            ]
        },

        # ==============================================================================
        # TFT ARCHITECTURE
        # ==============================================================================
        # hidden_size: Main hidden dimension throughout TFT
        # - Controls capacity of GRNs, attention, and LSTM layers
        # - Must be divisible by num_attention_heads
        # - Range 64-128 appropriate for ~200 series (avoids overfitting)
        # Compatibility check: hidden_size / nhead >= 16 for stable attention
        #   64/2=32✓, 64/4=16✓, 96/2=48✓, 96/4=24✓, 128/2=64✓, 128/4=32✓
        "hidden_size": {"values": [64, 96, 128]},

        # lstm_layers: Number of LSTM layers in encoder/decoder
        # - TFT's attention mechanism does heavy temporal lifting
        # - LSTM provides local sequential processing
        # - 1-2 layers sufficient; deeper adds parameters without proportional benefit
        "lstm_layers": {"values": [1, 2]},

        # num_attention_heads: Multi-head attention parallelism
        # - More heads = more diverse temporal pattern learning
        # - Limited to [2, 4] to ensure head_dim >= 16 with all hidden_size options
        # - Each head learns different "what to attend to" patterns
        "num_attention_heads": {"values": [2, 4]},

        # hidden_continuous_size: Dimension for continuous variable processing
        # - Separate from hidden_size for the variable selection networks
        # - With 50+ features, 16-48 provides adequate representation
        # - Larger values help capture feature interactions
        "hidden_continuous_size": {"values": [16, 32, 48]},

        # dropout: Dropout rate throughout TFT
        # - LOW values (0.05-0.15) for scarce signal
        # - High dropout would suppress neurons that learned rare conflict patterns
        # - Applied in GRNs, attention, and LSTM layers
        "dropout": {"values": [0.05, 0.15]},

        # full_attention: Whether to use full attention vs sparse/efficient variants
        # - True: Standard O(n²) attention, maximum expressiveness
        # - False: May use more efficient attention (implementation-dependent)
        # Worth exploring both for 36-60 timestep sequences
        "full_attention": {"values": [True, False]},

        # feed_forward: Feedforward network type in transformer layers
        # - GatedResidualNetwork: TFT's native architecture, well-tested
        # - SwiGLU: Modern gated activation (LLaMA, PaLM), smoother gradients
        # - GEGLU: GELU-based gating, strong empirical performance
        "feed_forward": {"values": ["GatedResidualNetwork", "SwiGLU"]},

        # add_relative_index: Add relative time index as a feature
        # - Helps model understand temporal position within the sequence
        # - Especially useful for capturing seasonality and trends
        "add_relative_index": {"values": [True]},

        # skip_interpolation: Skip interpolation in temporal processing
        # - False: Use interpolation for smoother representations
        # - True: Direct processing, may preserve sharp transitions better
        "skip_interpolation": {"values": [False, True]},

        # use_static_covariates: Whether to use static (time-invariant) features
        # - True: Leverages country-level constants (geography, etc.)
        # - False: Simpler model, may generalize better if static features are noisy
        "use_static_covariates": {"values": [False, True]},

        # norm_type: Normalization layer type
        # - RMSNorm: Faster, similar performance to LayerNorm (used in LLaMA)
        # - LayerNorm: Standard, more thoroughly tested
        "norm_type": {"values": ["RMSNorm"]},

        # use_reversible_instance_norm: Normalize per-instance before processing
        # - Helps with non-stationary data (conflict patterns change over time)
        # - "Reversible" means normalization stats are stored for inverse transform
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
        # - False Negative (non-zero→zero): weight = non_zero_weight × false_negative_weight (8-56)
        #
        # This encourages the model to:
        # 1. Learn strongly from actual conflict events (high non_zero_weight)
        # 2. Heavily penalize missing conflicts (high FN penalty)
        # 3. Be somewhat forgiving of false alarms (low FP weight encourages exploration)
        "loss_function": {"values": ["WeightedPenaltyHuberLoss"]},

        # zero_threshold: Scaled value below which predictions count as "zero"
        # - After AsinhTransform->MinMaxScaler, 1 fatality ≈ 0.11
        # - Range 0.08-0.23 spans 0-5 fatalities threshold and allows some margin for uncertainty
        # - Lower threshold = stricter zero classification
        "zero_threshold": {"values": [1e-4]},

        # delta: Huber loss transition point (L2 inside, L1 outside)
        # - Range 0.8-1.0 gives nearly pure L2 behavior for [0,1] scaled data
        # - Full L2 maximizes gradient signal from every error
        # - Important for learning from rare spikes where every gradient counts
        "delta": {
            "distribution": "uniform",
            "min": 0.3,
            "max": 1.0,
        },

        # non_zero_weight: Multiplier for non-zero actual values
        # - Fixed at 5.0 to reduce search dimensions
        # - Conflicts contribute 5x more to loss than zeros (counteracts class imbalance)
        # - FP and FN weights are tuned relative to this baseline
        "non_zero_weight": {"distribution": "uniform", "min": 10.0, "max": 50.0},
        # false_positive_weight: Multiplier when predicting non-zero for actual zero
        # - Range 0.5-1.0 (at or below baseline)
        # - Values <1.0 encourage model to "explore" non-zero predictions
        # - Helps escape local minimum of predicting all zeros
        "false_positive_weight": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1.5,
        },

        # false_negative_weight: Additional multiplier for missing actual conflicts
        # - Applied ON TOP of non_zero_weight: total FN penalty = non_zero × fn_weight
        # - Range 2-8 gives total FN weight of 8-56x baseline
        # - Highest penalty because missing conflicts is operationally costly
        "false_negative_weight": {
            "distribution": "uniform",
            "min": 3.0,
            "max": 10.0,
        },
    }

    sweep_config["parameters"] = parameters
    return sweep_config