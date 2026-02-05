def get_sweep_config():
    """
    N-HiTS (Neural Hierarchical Interpolation for Time Series) Sweep Configuration
    ================================================================================

    Problem Characteristics:
    ------------------------
    - ~200 time series (countries)
    - Zero-inflated target: Most country-months have zero fatalities
    - Heavy right skew: When conflicts occur, fatality counts vary enormously
    - Rare signal: Model must learn maximally from scarce non-zero events
    - 36-month forecast horizon with monthly resolution

    N-HiTS Architecture Overview:
    -----------------------------
    N-HiTS is designed for efficient long-horizon forecasting through:

    1. MULTI-RATE INPUT SAMPLING (Pooling):
       - Each stack applies different pooling kernel sizes to the input
       - Creates multi-resolution views of the time series
       - Stack 1: Large kernels → captures slow trends
       - Stack 2: Small kernels → captures fast dynamics/seasonality
       - Reduces computational cost vs processing full resolution everywhere

    2. HIERARCHICAL INTERPOLATION:
       - Each stack produces forecasts at different temporal resolutions
       - Lower resolution forecasts capture trends
       - Higher resolution forecasts add fine details
       - Interpolation combines them into final forecast
       - n_freq_downsample controls the resolution hierarchy

    3. STACK-BASED ARCHITECTURE (like N-BEATS):
       - Multiple stacks, each with multiple blocks
       - Each block: FC layers → basis expansion → forecast + backcast
       - Residual connections between blocks
       - Stacks specialize in different frequency components

    Key Design Decisions:
    ---------------------
    1. SCALING: AsinhTransform->MinMaxScaler for target
       - Asinh handles zeros naturally (unlike log)
       - MinMax bounds output to [0,1] for stable gradients
       - 1 fatality → ~0.11 after transform

    2. LOSS: WeightedPenaltyHuberLoss with high delta (0.8-1.0)
       - Full L2 behavior maximizes gradient signal from rare spikes
       - Asymmetric weights prioritize learning from actual conflicts

    3. POOLING: MaxPool vs AvgPool exploration
       - MaxPool preserves spike magnitudes (good for conflict detection)
       - AvgPool smoother but may dilute sparse events

    4. ARCHITECTURE: 2-3 stacks, moderate width
       - 2 stacks: Trend + seasonality decomposition
       - 3 stacks: Additional intermediate frequency component
       - Width 64-256 appropriate for ~200 series

    Hyperband Early Termination:
    ----------------------------
    - min_iter=20: Gives models time to find patterns in scarce signal
    - eta=2: Moderately aggressive pruning (keeps top 50% each round)

    Search Space Size: ~27,000 discrete combinations + continuous parameters
    Estimated sweep runs needed: 150-300 for good Bayesian coverage

    Returns:
        sweep_config (dict): WandB sweep configuration dictionary
    """

    sweep_config = {
        "method": "bayes",
        "name": "revolving_door_nhits_v4_mtd",
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
        # - N-HiTS handles shorter contexts efficiently due to pooling
        # - 36 months (3 years): Captures annual cycles, recent trends
        # - 48 months (4 years): Captures electoral cycles, medium-term patterns
        # Unlike attention-based models, N-HiTS doesn't need very long context
        # because pooling already extracts multi-scale patterns efficiently
        "input_chunk_length": {"values": [36, 48]},

        "output_chunk_shift": {"values": [0]},  # No gap between input and forecast
        "random_state": {"values": [67]},  # Reproducibility
        "mc_dropout": {"values": [True]},  # Monte Carlo dropout for uncertainty

        # ==============================================================================
        # N-HiTS ARCHITECTURE
        # ==============================================================================
        # num_stacks: Number of stacks (each captures different frequency range)
        # - 2 stacks: Classic trend + seasonality decomposition
        #   * Stack 1: Low frequency (trends, slow changes)
        #   * Stack 2: High frequency (seasonality, rapid dynamics)
        # - 3 stacks: Adds intermediate frequency component
        #   * May help capture conflict escalation patterns
        # - 1 stack: Loses multi-scale benefit (not recommended)
        # - 4+ stacks: Diminishing returns, overfitting risk for ~200 series
        "num_stacks": {"values": [2, 3]},

        # num_blocks: Blocks per stack (depth within each frequency scale)
        # - Each block processes residuals from previous block
        # - More blocks = more capacity to model complex patterns
        # - 1-3 blocks typically sufficient; more risks overfitting
        # - N-HiTS paper uses 1 block per stack as default
        "num_blocks": {"values": [1, 2, 3]},

        # num_layers: Fully connected layers per block
        # - Each block contains FC layers before basis expansion
        # - 2 layers: Simple, fast, less overfitting risk
        # - 3 layers: More capacity for complex nonlinear patterns
        # - 4+ layers: Rarely helps for time series; adds parameters
        "num_layers": {"values": [2, 3]},

        # layer_width: Hidden dimension of FC layers
        # - Controls capacity of each block's transformation
        # - 64: Conservative, fast, good for simple patterns
        # - 128: Balanced capacity and regularization
        # - 256: Higher capacity, may help capture complex conflict dynamics
        # - For ~200 series, avoid very wide (512+) to prevent overfitting
        "layer_width": {"values": [64, 128, 256]},

        # pooling_kernel_sizes: Controls multi-rate input sampling per stack
        # - None: Auto-configured based on input_chunk_length (RECOMMENDED)
        #   * Darts automatically creates geometric progression
        #   * Stack 1 gets largest kernels (captures slow trends)
        #   * Stack N gets smallest kernels (captures fast dynamics)
        # - Custom: tuple of tuples, shape (num_stacks, num_blocks)
        #   * Requires careful tuning; auto usually works well
        "pooling_kernel_sizes": {"values": [None]},

        # n_freq_downsample: Controls hierarchical interpolation resolution
        # - None: Auto-configured based on output_chunk_length (RECOMMENDED)
        #   * Creates geometric progression of downsampling factors
        #   * Stack 1 outputs lowest resolution (coarse trends)
        #   * Stack N outputs highest resolution (fine details)
        # - Custom: tuple matching num_stacks
        #   * Requires careful tuning; auto usually works well
        "n_freq_downsample": {"values": [None]},

        # max_pool_1d: Pooling type for multi-rate sampling
        # - True (MaxPool): Preserves peak values in each pooling window
        #   * Better for sparse event detection (conflict spikes)
        #   * Retains maximum signal even when most timesteps are zero
        # - False (AvgPool): Averages values in pooling window
        #   * Smoother, may miss isolated spikes in sparse data
        #   * Better for dense, continuous patterns
        # For zero-inflated conflict data, MaxPool likely better
        "max_pool_1d": {"values": [True, False]},

        # activation: Non-linearity between FC layers
        # - ReLU: Fast, sparse activations, good default
        #   * May cause "dead neurons" but rarely an issue
        # - GELU: Smooth approximation to ReLU, used in transformers
        #   * Often slightly better for time series
        #   * Marginally slower than ReLU
        # - ELU/SiLU: Alternatives with minimal practical difference
        "activation": {"values": ["ReLU", "GELU"]},

        # dropout: Regularization within FC blocks
        # - Applied between FC layers
        # - LOW values (0.05-0.15) for scarce signal
        # - High dropout would suppress neurons learning rare conflict patterns
        # - N-HiTS already has implicit regularization from pooling
        "dropout": {"values": [0.05, 0.15]},

        # ==============================================================================
        # TRAINING BASICS
        # ==============================================================================
        # batch_size: Samples per gradient update
        # - Larger batches help zero-inflated data see more non-zero events
        # - N-HiTS is computationally lighter than attention models
        # - Range 256-2048 balances GPU utilization with gradient quality
        "batch_size": {"values": [256, 512, 1024, 2048]},

        # n_epochs: Maximum training epochs
        # - N-HiTS typically converges faster than attention models
        # - 150 epochs provides headroom for scarce signal learning
        # - Early stopping will trigger before max in most cases
        "n_epochs": {"values": [150]},

        # early_stopping_patience: Epochs without improvement before stopping
        # - Higher patience (15-25) for scarce signal
        # - Rare conflict patterns may take time to emerge in validation
        "early_stopping_patience": {"values": [20]},

        # early_stopping_min_delta: Minimum improvement to count as progress
        # - Small values appropriate for [0,1] scaled loss
        "early_stopping_min_delta": {"values": [0.0001]},

        "force_reset": {"values": [True]},  # Clean model state each sweep run

        # ==============================================================================
        # OPTIMIZER / LEARNING RATE SCHEDULE
        # ==============================================================================
        # lr: Learning rate (log-uniform for proper exploration)
        # - 5e-5: Conservative, stable learning
        # - 1e-3: Aggressive, faster but risk of instability
        # - N-HiTS typically works well in 1e-4 to 5e-4 range
        "lr": {
            "distribution": "log_uniform_values",
            "min": 5e-5,
            "max": 1e-3,
        },

        # weight_decay: L2 regularization
        # DISABLED because:
        # - Scarce signal means every neuron learning rare patterns is precious
        # - N-HiTS already regularized by pooling structure
        # - Previous experiments showed weight decay hurts rare event learning
        "weight_decay": {"values": [0]},

        # lr_scheduler: ReduceLROnPlateau configuration
        "lr_scheduler_factor": {"values": [0.5]},  # Halve LR when stuck
        "lr_scheduler_patience": {"values": [8]},  # Wait 8 epochs before reducing
        "lr_scheduler_min_lr": {"values": [1e-6]},  # Floor prevents negligible LR

        # gradient_clip_val: Maximum gradient norm
        # - Prevents exploding gradients
        # - N-HiTS has stable gradients due to simple FC architecture
        # - Range 0.5-1.5 is conservative
        "gradient_clip_val": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1.5,
        },

        # ==============================================================================
        # INSTANCE NORMALIZATION
        # ==============================================================================
        # use_reversible_instance_norm: Per-instance normalization
        # - Normalizes each time series independently before processing
        # - "Reversible" stores stats to invert normalization on output
        # - True: Helps with non-stationary data (conflict patterns evolve)
        # - False: Simpler, may generalize better if series are comparable
        # Worth exploring both for conflict data
        "use_reversible_instance_norm": {"values": [True, False]},

        # ==============================================================================
        # LOSS FUNCTION: WeightedPenaltyHuberLoss
        # ==============================================================================
        # Custom loss for zero-inflated rare event forecasting.
        #
        # Weight multiplication logic:
        # - True Negative (zero→zero): weight = 1.0 (baseline)
        # - False Positive (zero→non-zero): weight = false_positive_weight (0.5-1.0)
        # - True Positive (non-zero→non-zero): weight = non_zero_weight (4-7)
        # - False Negative (non-zero→zero): weight = non_zero_weight × fn_weight (8-56)
        #
        # This encourages:
        # 1. Strong learning from actual conflict events (high non_zero_weight)
        # 2. Heavy penalty for missing conflicts (high FN penalty)
        # 3. Tolerance for false alarms (low FP weight encourages exploration)
        "loss_function": {"values": ["WeightedPenaltyHuberLoss"]},

        # zero_threshold: Scaled value below which predictions count as "zero"
        # - After AsinhTransform->MinMaxScaler, 1 fatality ≈ 0.11
        # - Range 0.11-0.20 spans 1 fatality threshold and allows some margin for uncertainty
        # - Lower threshold = stricter zero classification
        "zero_threshold": {
            "distribution": "log_uniform_values",
            "min": 0.11, # 2 fatalities threshold after scaling
            "max": 0.23, # 5 fatalities threshold after scaling
        },
        # delta: Huber loss transition point
        # - Range 0.8-1.0: Nearly pure L2 for [0,1] scaled data
        # - Full L2 maximizes gradient signal from rare spikes
        "delta": {
            "distribution": "uniform",
            "min": 0.8,
            "max": 1.0,
        },

        # non_zero_weight: Multiplier for non-zero actual values
        # - Range 4-7: Conflict events contribute 4-7x more to loss
        # - Counteracts class imbalance (mostly zeros)
        "non_zero_weight": {
            "distribution": "uniform",
            "min": 4.0,
            "max": 7.0,
        },

        # false_positive_weight: Penalty for predicting conflict when none occurred
        # - Range 0.5-1.0: At or below baseline
        # - Values <1.0 encourage model to explore non-zero predictions
        # - Helps escape local minimum of predicting all zeros
        "false_positive_weight": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1.0,
        },

        # false_negative_weight: Additional penalty for missing actual conflicts
        # - Applied on top of non_zero_weight: FN = non_zero × fn_weight
        # - Range 2-8: Total FN penalty of 8-56x baseline
        # - Highest penalty: missing conflicts is operationally costly
        "false_negative_weight": {
            "distribution": "uniform",
            "min": 2.0,
            "max": 8.0,
        },

        # ==============================================================================
        # FEATURE SCALING
        # ==============================================================================
        # feature_scaler: Global default (None = use feature_scaler_map)
        "feature_scaler": {"values": [None]},

        # target_scaler: AsinhTransform->MinMaxScaler
        # - Asinh: Handles zeros, compresses extremes
        #   * asinh(0) = 0 (no log(0) issues)
        #   * Linear near 0, logarithmic for large values
        # - MinMax: Bounds to [0,1] for stable gradients
        "target_scaler": {"values": ["AsinhTransform->MinMaxScaler"]},

        # feature_scaler_map: Per-feature scaling based on distribution
        "feature_scaler_map": {
            "values": [
                {
                    # AsinhTransform->MinMaxScaler: Zero-inflated and right-skewed
                    # Features with many zeros and occasional extreme values
                    "AsinhTransform->MinMaxScaler": [
                        # Conflict fatality counts (zero-inflated, extreme outliers)
                        "lr_ged_sb", "lr_ged_ns", "lr_ged_os",
                        "lr_acled_sb", "lr_acled_sb_count", "lr_acled_os",
                        "lr_ged_sb_tsum_24",  # 24-month cumulative
                        "lr_splag_1_ged_sb", "lr_splag_1_ged_os", "lr_splag_1_ged_ns",
                        # Economic indicators (GDP spans orders of magnitude)
                        "lr_wdi_ny_gdp_mktp_kd", "lr_wdi_nv_agr_totl_kn",
                        "lr_wdi_sm_pop_netm", "lr_wdi_sm_pop_refg_or",
                        # Mortality rates (positive, skewed)
                        "lr_wdi_sp_dyn_imrt_fe_in",
                        # Token counts from text (zero-inflated)
                        "lr_topic_tokens_t1", "lr_topic_tokens_t1_splag",
                    ],
                    # MinMaxScaler: Bounded or roughly symmetric features
                    # Already in reasonable ranges, just need [0,1] normalization
                    "MinMaxScaler": [
                        # WDI percentages (0-100 scale)
                        "lr_wdi_sl_tlf_totl_fe_zs", "lr_wdi_se_enr_prim_fm_zs",
                        "lr_wdi_sp_urb_totl_in_zs", "lr_wdi_sh_sta_maln_zs",
                        "lr_wdi_sh_sta_stnt_zs", "lr_wdi_dt_oda_odat_pc_zs",
                        "lr_wdi_ms_mil_xpnd_gd_zs",
                        # V-Dem indices (already 0-1)
                        "lr_vdem_v2x_horacc", "lr_vdem_v2xnp_client", "lr_vdem_v2x_veracc",
                        "lr_vdem_v2x_divparctrl", "lr_vdem_v2xpe_exlpol", "lr_vdem_v2x_diagacc",
                        "lr_vdem_v2xpe_exlgeo", "lr_vdem_v2xpe_exlgender", "lr_vdem_v2xpe_exlsocgr",
                        "lr_vdem_v2x_ex_party", "lr_vdem_v2x_genpp", "lr_vdem_v2xeg_eqdr",
                        "lr_vdem_v2xcl_prpty", "lr_vdem_v2xeg_eqprotec", "lr_vdem_v2x_ex_military",
                        "lr_vdem_v2xcl_dmove", "lr_vdem_v2x_clphy", "lr_vdem_v2x_hosabort",
                        "lr_vdem_v2xnp_regcorr",
                        # Topic thetas (probability proportions, sum to ~1)
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
                        # Population growth (small range, can be negative)
                        "lr_wdi_sp_pop_grow",
                    ],
                }
            ]
        },
    }

    sweep_config["parameters"] = parameters
    return sweep_config