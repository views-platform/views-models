def get_sweep_config():
    """
    Transformer Model Hyperparameter Sweep Configuration
    ======================================================

    Problem Characteristics:
    ------------------------
    - ~200 time series (countries)
    - Zero-inflated target: Most country-months have zero fatalities
    - Heavy right skew: When conflicts occur, fatality counts vary enormously
    - Rare signal: Model must learn maximally from scarce non-zero events
    - 36-month forecast horizon with monthly resolution

    Transformer Architecture Strengths:
    ------------------------------------
    - Self-attention captures long-range temporal dependencies without recurrence
    - Multi-head attention learns diverse temporal patterns in parallel
    - Positional encoding preserves sequential information
    - Encoder-decoder structure separates pattern extraction from forecasting
    - Parallelizable training (faster than RNNs)

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
       - d_model/nhead >= 32 required for stable attention
       - GLU activations (SwiGLU, GEGLU) outperform standard activations

    Attention Stability Constraint:
    --------------------------------
    d_model must be divisible by num_attention_heads, and the quotient
    (head dimension) should be >= 32 for stable attention computation.

    Valid combinations in this config:
    - d_model=64:  nhead=2 (head_dim=32✓)
    - d_model=128: nhead=2 (64✓), nhead=4 (32✓)
    - d_model=256: nhead=2 (128✓), nhead=4 (64✓), nhead=8 (32✓)

    Invalid combinations (will cause issues):
    - d_model=64, nhead=4 (head_dim=16 < 32)
    - d_model=64, nhead=8 (head_dim=8 < 32)
    - d_model=128, nhead=8 (head_dim=16 < 32)

    Note: Darts may handle some invalid combinations gracefully, but attention
    quality degrades with small head dimensions.

    Hyperband Early Termination:
    ----------------------------
    - min_iter=20: Gives transformers time to find patterns (they converge slower)
    - eta=2: Moderately aggressive pruning (keeps top 50% each round)

    Search Space Size: ~93,000 discrete combinations + continuous parameters
    Estimated sweep runs needed: 300-500 for good Bayesian coverage

    Returns:
        sweep_config (dict): WandB sweep configuration dictionary
    """

    sweep_config = {
        "method": "bayes",
        "name": "good_life_transformer_v7_mtd",
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
        # - Transformers excel with longer context (attention has no recurrence decay)
        # - 48 months (4 years): Captures electoral cycles, medium-term trends
        # - 60 months (5 years): Captures longer political/economic cycles
        # - 72 months (6 years): Maximum context for deep historical patterns
        # Attention mechanism can selectively focus on relevant past periods
        "input_chunk_length": {"values": [48, 60, 72]},

        "output_chunk_length": {"values": [36]},  # Must match steps
        "output_chunk_shift": {"values": [0]},  # No gap between input and forecast
        "mc_dropout": {"values": [True]},  # Monte Carlo dropout for uncertainty
        "random_state": {"values": [67]},  # Reproducibility
        "detect_anomaly": {"values": [False]},  # Only for debugging (slows training)

        # ==============================================================================
        # TRAINING BASICS
        # ==============================================================================
        # batch_size: Samples per gradient update
        # - Larger batches help zero-inflated data see more non-zero events
        # - Transformers are memory-efficient (parallel attention)
        # - Range 256-2048 balances GPU memory with gradient quality
        "batch_size": {"values": [512, 1024, 2048, 4096]},

        # n_epochs: Maximum training epochs
        # - Transformers often need more epochs than RNNs to converge
        # - 150 epochs provides headroom; early stopping triggers before max
        "n_epochs": {"values": [150]},

        # early_stopping_patience: Epochs without improvement before stopping
        # - Higher patience (15-25) for scarce signal
        # - Transformers may plateau then improve as attention patterns refine
        "early_stopping_patience": {"values": [20]},

        # early_stopping_min_delta: Minimum improvement to count as progress
        # - Small values (5e-5 to 1e-4) appropriate for [0,1] scaled loss
        "early_stopping_min_delta": {"values": [0.0001]},

        "force_reset": {"values": [True]},  # Clean model state each sweep run

        # ==============================================================================
        # OPTIMIZER / LEARNING RATE SCHEDULE
        # ==============================================================================
        # lr: Learning rate (log-uniform distribution for proper exploration)
        # - Transformers are sensitive to LR; too high causes attention instability
        # - 5e-5: Conservative, stable but slow learning
        # - 1e-3: Aggressive upper bound
        # - Optimal typically in 1e-4 to 5e-4 range for time series
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
        # - Dropout provides sufficient regularization for transformers
        "weight_decay": {"values": [0]},

        # lr_scheduler: ReduceLROnPlateau configuration
        # - factor=0.5: Halve LR when stuck (standard, well-tested)
        # - patience=8: Wait 8 epochs before reducing (allows temporary plateaus)
        # - min_lr=1e-6: Floor prevents LR from becoming negligible
        "lr_scheduler_factor": {"values": [0.5]},
        "lr_scheduler_patience": {"values": [8]},
        "lr_scheduler_min_lr": {"values": [1e-6]},

        # gradient_clip_val: Maximum gradient norm
        # - Prevents exploding gradients in attention layers
        # - Transformers generally have stable gradients but clipping helps
        # - Range 0.5-1.5 is conservative for scaled [0,1] data
        "gradient_clip_val": {"values": [1.5]},
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
        # TRANSFORMER ARCHITECTURE
        # ==============================================================================
        # d_model: Embedding dimension throughout the transformer
        # - Controls capacity of attention and feedforward layers
        # - Must be divisible by num_attention_heads
        # - 64: Lightweight, fast, good for simple patterns
        # - 128: Balanced capacity for moderate complexity
        # - 256: Higher capacity for complex temporal dependencies
        # For ~200 series, avoid very large (512+) to prevent overfitting
        "d_model": {"values": [128, 256]},

        # num_attention_heads: Parallel attention mechanisms
        # - Each head learns different "what to attend to" patterns
        # - More heads = more diverse temporal pattern learning
        # - Constraint: d_model / nhead >= 32 for stable attention
        # - 2 heads: Simple, interpretable attention
        # - 4 heads: Good balance of diversity and stability
        # - 8 heads: Maximum diversity (only valid with d_model >= 256)
        "num_attention_heads": {"values": [2, 4, 8]},

        # num_encoder_layers: Depth of pattern extraction
        # - Encoder processes input sequence to extract representations
        # - 1 layer: Simple patterns, fast, less overfitting
        # - 2 layers: Moderate depth for hierarchical patterns
        # - 3 layers: Deeper feature extraction (may overfit with scarce signal)
        "num_encoder_layers": {"values": [1, 2, 3]},

        # num_decoder_layers: Depth of forecast generation
        # - Decoder generates output sequence from encoder representations
        # - Typically shallower than encoder (forecasting is "simpler" than encoding)
        # - 1-2 layers usually sufficient for time series
        "num_decoder_layers": {"values": [1, 2]},

        # dim_feedforward: Hidden dimension of feedforward networks
        # - FFN in each transformer layer: d_model → dim_ff → d_model
        # - Standard practice: 2-4x d_model
        # - 256: Conservative (2x of d_model=128)
        # - 512: Balanced (4x of d_model=128, 2x of d_model=256)
        # - 1024: Higher capacity (4x of d_model=256)
        "dim_feedforward": {"values": [256, 512, 1024]},

        # dropout: Regularization throughout transformer
        # - Applied in attention, feedforward, and embeddings
        # - LOW values (0.05-0.15) for scarce signal
        # - High dropout would suppress neurons learning rare conflict patterns
        "dropout": {"values": [0.05, 0.15]},

        # activation: Feedforward network activation function
        # - SwiGLU: Gated activation used in LLaMA, PaLM (best empirical results)
        #   * Swish(xW) ⊙ (xV) - smooth gating mechanism
        # - GEGLU: GELU-based gating, strong alternative
        #   * GELU(xW) ⊙ (xV)
        # - gelu: Standard GELU without gating (simpler, fewer parameters)
        # GLU variants add ~50% parameters but typically improve performance
        "activation": {"values": ["SwiGLU", "GEGLU", "gelu"]},

        # norm_type: Normalization layer type
        # - RMSNorm: Root Mean Square normalization (used in LLaMA)
        #   * Faster than LayerNorm (no mean computation)
        #   * Similar performance in practice
        # - LayerNorm: Standard layer normalization
        #   * More thoroughly tested, slight overhead
        "norm_type": {"values": ["RMSNorm", "LayerNorm"]},

        # use_reversible_instance_norm: Per-instance normalization
        # - Normalizes each time series independently before processing
        # - "Reversible" stores stats to invert normalization on output
        # - True: Helps with non-stationary data (conflict patterns evolve)
        # - False: Simpler, may generalize better if series are comparable
        "use_reversible_instance_norm": {"values": [False, True]},

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
        "loss_function": {"values": ["WeightedPenaltyHuberLoss"]},

        # zero_threshold: Scaled value below which predictions count as "zero"
        # - After AsinhTransform->MinMaxScaler, 1 fatality ≈ 0.11
        # - Range 0.08-0.23 spans 0-5 fatalities threshold and allows some margin for uncertainty
        # - Lower threshold = stricter zero classification
        "zero_threshold": {
            "distribution": "uniform",
            "min": 0.05,
            "max": 0.23,
        },

        # delta: Huber loss transition point (L2 inside delta, L1 outside)
        # - Range 0.8-1.0 gives nearly pure L2 behavior for [0,1] scaled data
        # - Full L2 maximizes gradient signal from every error
        # - Important for learning from rare spikes where every gradient counts
        "delta": {
            "distribution": "uniform",
            "min": 0.70,
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
            "max": 2,
        },

        # false_negative_weight: Additional penalty for missing actual conflicts
        # - Applied on top of non_zero_weight: FN = non_zero × fn_weight
        # - Range 2-8: Total FN penalty of 8-56x baseline
        # - Highest penalty: missing conflicts is operationally costly
        "false_negative_weight": {
            "distribution": "uniform",
            "min": 1.0,
            "max": 6.0,
        },
    }

    sweep_config["parameters"] = parameters
    return sweep_config