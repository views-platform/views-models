def get_sweep_config():
    """
    N-BEATS (Neural Basis Expansion Analysis for Time Series) Sweep Configuration
    ===============================================================================
    Optimized for country-month level rare conflict fatality forecasting.

    Problem Characteristics:
    ------------------------
    - ~200 time series (countries)
    - Zero-inflated target: Most country-months have zero fatalities
    - Heavy right skew: When conflicts occur, fatality counts vary enormously
    - Rare signal: Model must learn maximally from scarce non-zero events
    - 36-month forecast horizon with monthly resolution

    N-BEATS Architecture Overview:
    ------------------------------
    N-BEATS is a pure deep learning approach using stacked fully-connected blocks:

    1. STACK-BASED ARCHITECTURE:
       - Multiple stacks, each with multiple blocks
       - Each block: FC layers → basis expansion → forecast + backcast
       - Backcast subtracts learned patterns from input (residual learning)
       - Stacks process progressively refined residuals

    2. TWO ARCHITECTURE MODES:
       - Generic (generic_architecture=True):
         * Learnable basis functions (fully flexible)
         * Better for irregular, unpredictable patterns
         * More parameters, higher capacity
       - Interpretable (generic_architecture=False):
         * Trend + Seasonality decomposition
         * Explicit polynomial (trend) and Fourier (seasonality) bases
         * More constrained, potentially better generalization
         * Provides interpretable components

    3. DOUBLY RESIDUAL DESIGN:
       - Block-level residuals (backcast subtraction)
       - Stack-level residuals
       - Ensures gradient flow even in deep networks

    Key Design Decisions:
    ---------------------
    1. SCALING: AsinhTransform->MinMaxScaler for target
       - Asinh handles zeros naturally (unlike log)
       - MinMax bounds output to [0,1] for stable gradients
       - 1 fatality → ~0.11 after transform

    2. LOSS: WeightedPenaltyHuberLoss with high delta (0.8-1.0)
       - Full L2 behavior maximizes gradient signal from rare spikes
       - Asymmetric weights prioritize learning from actual conflicts

    3. REGULARIZATION: LOW dropout (0.05-0.15)
       - N-BEATS is prone to overfitting but scarce signal needs preservation
       - Low dropout retains neurons that learn rare patterns
       - weight_decay=0 to prevent weight collapse

    4. ARCHITECTURE: Moderate depth and width
       - 2-3 stacks for multi-scale pattern learning
       - 1-3 blocks per stack
       - Width 64-256 appropriate for ~200 series

    Generic vs Interpretable:
    -------------------------
    For conflict forecasting, both modes are worth exploring:
    - Generic: May capture irregular conflict dynamics better
    - Interpretable: Trend component may capture escalation; seasonality less relevant

    The interpretable mode's trend stack uses polynomial basis which can model
    slow escalation/de-escalation patterns in conflict.

    Hyperband Early Termination:
    ----------------------------
    - min_iter=20: Gives N-BEATS time to find patterns in scarce signal
    - eta=2: Moderately aggressive pruning (keeps top 50% each round)

    Search Space Size: ~5,200 discrete combinations + continuous parameters
    Estimated sweep runs needed: 100-150 for good Bayesian coverage

    Returns:
        sweep_config (dict): WandB sweep configuration dictionary
    """

    sweep_config = {
        "method": "bayes",
        "name": "new_rules_nbeats_v3_mtd",
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
        # - N-BEATS processes the full input through FC layers
        # - 36 months (3 years): Captures annual cycles, recent trends
        # - 48 months (4 years): Captures electoral cycles, medium-term patterns
        # - N-BEATS is efficient with moderate sequence lengths
        "input_chunk_length": {"values": [48, 60, 72]},
        "output_chunk_length": {"values": [36]},  # Must match steps
        "output_chunk_shift": {"values": [0]},  # No gap between input and forecast
        "mc_dropout": {"values": [True]},  # Monte Carlo dropout for uncertainty
        "random_state": {"values": [67]},  # Reproducibility
        # ==============================================================================
        # TRAINING BASICS
        # ==============================================================================
        # batch_size: Samples per gradient update
        # - Larger batches help zero-inflated data see more non-zero events
        # - N-BEATS is computationally efficient (just FC layers)
        # - Range 1024-4096 for stable gradients with sparse signal
        "batch_size": {"values": [1024, 2048, 4096]},
        # n_epochs: Maximum training epochs
        # - 150 epochs provides headroom; early stopping triggers before max
        "n_epochs": {"values": [150]},
        # early_stopping_patience: Epochs without improvement before stopping
        # - Fixed to 20 for scarce signal (need time to find patterns)
        "early_stopping_patience": {"values": [20]},
        # early_stopping_min_delta: Minimum improvement to count as progress
        # - Small value appropriate for [0,1] scaled loss
        "early_stopping_min_delta": {"values": [0.0001]},
        "force_reset": {"values": [True]},  # Clean model state each sweep run
        # ==============================================================================
        # OPTIMIZER / LEARNING RATE SCHEDULE
        # ==============================================================================
        # lr: Learning rate (log-uniform distribution for proper exploration)
        # - N-BEATS can be sensitive to LR due to deep stacking
        # - 5e-5: Conservative, stable learning
        # - 1e-3: Aggressive upper bound
        # - Optimal typically in 1e-4 to 5e-4 range
        "lr": {
            "distribution": "log_uniform_values",
            "min": 5e-5,
            "max": 1e-3,
        },
        # weight_decay: L2 regularization on weights
        # DISABLED (set to 0) because:
        # - Scarce signal means neurons learning rare patterns are precious
        # - Previous experiments showed weight collapse with weight_decay > 0
        # - Dropout provides regularization; weight_decay would be excessive
        "weight_decay": {"values": [0]},
        # lr_scheduler: ReduceLROnPlateau configuration
        # - factor=0.5: Halve LR when stuck
        # - patience=8: Wait 8 epochs before reducing
        # - min_lr=1e-6: Floor prevents LR from becoming negligible
        "lr_scheduler_factor": {"values": [0.5]},
        "lr_scheduler_patience": {"values": [8]},
        "lr_scheduler_min_lr": {"values": [1e-6]},
        # gradient_clip_val: Maximum gradient norm
        # - Prevents exploding gradients in deep stacks
        # - N-BEATS has residual connections which help stability
        # - Range 0.5-1.5 is conservative for [0,1] scaled data
        "gradient_clip_val": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1.0,
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
        #   * Calibration: asinh(1)≈0.88 → after MinMax ≈0.11
        "target_scaler": {"values": ["AsinhTransform->MinMaxScaler"]},
        # feature_scaler_map: Per-feature scaling based on distribution characteristics
        "feature_scaler_map": {
            "values": [
                {
                    # AsinhTransform->MinMaxScaler: For zero-inflated and right-skewed features
                    "AsinhTransform->MinMaxScaler": [
                        # Conflict fatality counts (zero-inflated, extreme outliers)
                        "lr_ged_sb",
                        "lr_ged_ns",
                        "lr_ged_os",
                        "lr_acled_sb",
                        "lr_acled_sb_count",
                        "lr_acled_os",
                        "lr_ged_sb_tsum_24",
                        "lr_splag_1_ged_sb",
                        "lr_splag_1_ged_os",
                        "lr_splag_1_ged_ns",
                        # Economic indicators with extreme skew
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
                        # Topic model theta values (probability distributions)
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
                        # Population growth rate
                        "lr_wdi_sp_pop_grow",
                    ],
                }
            ]
        },
        # ==============================================================================
        # N-BEATS ARCHITECTURE
        # ==============================================================================
        # generic_architecture: Architecture mode selection
        # - True (Generic): Learnable basis functions
        #   * Fully flexible, can learn any pattern
        #   * More parameters, higher capacity
        #   * May capture irregular conflict dynamics better
        # - False (Interpretable): Trend + Seasonality decomposition
        #   * Polynomial basis for trend, Fourier for seasonality
        #   * More constrained, potentially better generalization
        #   * Trend component may capture conflict escalation patterns
        # Worth exploring both for conflict forecasting
        "generic_architecture": {"values": [True]},
        # num_stacks: Number of stacks in the network
        # - Each stack processes residuals from previous stack
        # - 2 stacks: Simpler, less overfitting risk
        #   * Interpretable: 1 trend + 1 seasonality stack
        #   * Generic: 2 learnable pattern stacks
        # - 3 stacks: More capacity for complex patterns
        # - 4+ stacks: Diminishing returns, overfitting risk for ~200 series
        "num_stacks": {"values": [2, 3]},
        # num_blocks: Blocks per stack
        # - Each block produces forecast and backcast
        # - More blocks = more residual refinement within each stack
        # - 1 block: Simplest, N-BEATS paper default
        # - 2-3 blocks: More capacity per stack
        "num_blocks": {"values": [1, 2]},
        # num_layers: Fully connected layers per block
        # - Each block contains FC layers before basis expansion
        # - 2 layers: Simple, fast, less overfitting
        # - 3 layers: More nonlinearity for complex patterns
        "num_layers": {"values": [2, 3]},
        # layer_widths: Hidden dimension of FC layers
        # - Controls capacity of each block
        # - 64: Conservative, fast, less overfitting
        # - 128: Balanced for ~200 series
        # - 256: Higher capacity (monitor for overfitting)
        # - Avoid 512+ for ~200 series (overfitting risk)
        "layer_width": {"values": [64, 128, 256]},
        # activation: Non-linearity between FC layers
        # - ReLU: Fast, sparse activations, standard choice
        # - GELU: Smoother gradients, often slightly better for time series
        "activation": {"values": ["ReLU", "GELU"]},
        # dropout: Regularization within FC blocks
        # - LOW values (0.05-0.15) for scarce signal
        # - High dropout would suppress neurons learning rare conflict patterns
        # - N-BEATS is prone to overfitting, but signal preservation is priority
        "dropout": {"values": [0.05, 0.15]},
        # ==============================================================================
        # LOSS FUNCTION: WeightedPenaltyHuberLoss
        # ==============================================================================
        # Custom loss designed for zero-inflated rare event forecasting.
        #
        # Weight multiplication logic:
        # - True Negative (zero→zero): weight = 1.0 (baseline)
        # - False Positive (zero→non-zero): weight = false_positive_weight (0.5-1.0)
        # - True Positive (non-zero→non-zero): weight = non_zero_weight (4-7)
        # - False Negative (non-zero→zero): weight = non_zero_weight × fn_weight (8-56)
        "loss_function": {"values": ["WeightedPenaltyHuberLoss"]},
        # zero_threshold: Scaled value below which predictions count as "zero"
        # - After AsinhTransform->MinMaxScaler, 1 fatality ≈ 0.11
        # - Range 0.08-0.23 spans 0-5 fatalities threshold and allows some margin for uncertainty
        # - Lower threshold = stricter zero classification
        "zero_threshold": {"values": [1e-4]},
        # delta: Huber loss transition point (L2 inside delta, L1 outside)
        # - Range 0.8-1.0 gives nearly pure L2 behavior for [0,1] scaled data
        # - Full L2 maximizes gradient signal from every error
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
        # - Values <1.0 encourage model to explore non-zero predictions
        "false_positive_weight": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1.0,
        },
        # false_negative_weight: Additional multiplier for missing actual conflicts
        # - Applied ON TOP of non_zero_weight: total FN penalty = non_zero × fn_weight
        # - Range 2-8 gives total FN weight of 8-56x baseline
        "false_negative_weight": {
            "distribution": "uniform",
            "min": 1.0,
            "max": 3.0,
        },
    }

    sweep_config["parameters"] = parameters
    return sweep_config
