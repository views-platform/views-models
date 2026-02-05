def get_sweep_config():
    """
    BlockRNN (LSTM/GRU) Hyperparameter Sweep Configuration
    ========================================================

    Problem Characteristics:
    ------------------------
    - ~200 time series (countries)
    - Zero-inflated target: Most country-months have zero fatalities
    - Heavy right skew: When conflicts occur, fatality counts vary enormously
    - Rare signal: Model must learn maximally from scarce non-zero events
    - 36-month forecast horizon with monthly resolution

    BlockRNN Architecture Overview:
    -------------------------------
    BlockRNN wraps PyTorch's LSTM/GRU into a block-based forecasting model:

    1. RECURRENT PROCESSING:
       - Sequential processing of input time steps
       - Hidden state accumulates temporal information
       - Cell state (LSTM only) provides long-term memory

    2. LSTM vs GRU:
       - LSTM: Separate cell state and hidden state
         * Better long-term memory retention
         * More parameters (4 gates vs 3)
         * May help with conflict escalation patterns over years
       - GRU: Combined cell/hidden state
         * Faster training, fewer parameters
         * Often comparable performance to LSTM
         * Better for shorter-term patterns

    3. STACKED LAYERS:
       - Multiple RNN layers process hierarchically
       - Lower layers: Local patterns
       - Higher layers: Abstract representations
       - Dropout between layers for regularization

    Key Design Decisions:
    ---------------------
    1. SCALING: AsinhTransform->MinMaxScaler for target
       - Asinh handles zeros naturally (unlike log)
       - MinMax bounds output to [0,1] for stable gradients
       - 1 fatality → ~0.11 after transform

    2. LOSS: WeightedPenaltyHuberLoss with high delta (0.8-1.0)
       - Full L2 behavior maximizes gradient signal from rare spikes
       - Asymmetric weights prioritize learning from actual conflicts

    3. REGULARIZATION: Minimal (weight_decay=0, low dropout)
       - Scarce signal means preserving neurons that learn rare patterns
       - RNNs already have implicit regularization through recurrence

    4. ARCHITECTURE: Small hidden dims, shallow depth
       - ~200 series cannot support large hidden states (overfitting risk)
       - 1-3 layers sufficient; deeper RNNs have vanishing gradient issues
       - GRU often matches LSTM with fewer parameters

    RNN-Specific Considerations:
    ----------------------------
    - Gradient clipping: Critical for RNNs (exploding gradients through time)
    - Wider clipping range (0.5-2.0) than transformers to allow gradient flow
    - Hidden dim scales memory requirements quadratically

    Hyperband Early Termination:
    ----------------------------
    - min_iter=20: Gives RNNs time to learn temporal patterns
    - eta=2: Moderately aggressive pruning (keeps top 50% each round)

    Search Space Size: ~15,500 discrete combinations + continuous parameters
    Estimated sweep runs needed: 150-250 for good Bayesian coverage

    Returns:
        sweep_config (dict): WandB sweep configuration dictionary
    """

    sweep_config = {
        "method": "bayes",
        "name": "dancing_queen_blockrnn_v3_mtd",
        "early_terminate": {"type": "hyperband", "min_iter": 20, "eta": 2},
        "metric": {"name": "time_series_wise_mtd_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        # steps: 36-month forecast horizon (standard for conflict forecasting)
        "steps": {"values": [[*range(1, 36 + 1)]]},

        # input_chunk_length: Historical context window
        # - RNNs process sequences step-by-step, accumulating in hidden state
        # - 36 months (3 years): Captures annual cycles, recent escalation
        # - 48 months (4 years): Captures electoral cycles, medium-term patterns
        # - Longer sequences increase vanishing gradient risk but provide more context
        # - RNNs handle moderate lengths well; hidden state compresses history
        "input_chunk_length": {"values": [36, 48]},

        "output_chunk_shift": {"values": [0]},  # No gap between input and forecast
        "random_state": {"values": [67]},  # Reproducibility
        "mc_dropout": {"values": [True]},  # Monte Carlo dropout for uncertainty

        # ==============================================================================
        # TRAINING BASICS
        # ==============================================================================
        # batch_size: Samples per gradient update
        # - Larger batches help zero-inflated data see more non-zero events
        # - RNNs are memory-intensive (hidden states stored for BPTT)
        # - Range 512-2048 balances GPU memory with gradient quality
        # - Removed 4096 to avoid OOM with long sequences
        "batch_size": {"values": [512, 1024, 2048]},

        # n_epochs: Maximum training epochs
        # - RNNs may need more epochs than feedforward models
        # - 200 epochs provides headroom; early stopping triggers before max
        "n_epochs": {"values": [200]},

        # early_stopping_patience: Epochs without improvement before stopping
        # - Higher patience (15-25) for scarce signal
        # - RNN loss can be noisy; patience prevents premature stopping
        "early_stopping_patience": {"values": [20]},

        # early_stopping_min_delta: Minimum improvement to count as progress
        # - Small values (5e-5 to 1e-4) appropriate for [0,1] scaled loss
        "early_stopping_min_delta": {"values": [0.0001]},

        "force_reset": {"values": [True]},  # Clean model state each sweep run

        # ==============================================================================
        # OPTIMIZER / LEARNING RATE SCHEDULE
        # ==============================================================================
        # lr: Learning rate (log-uniform distribution for proper exploration)
        # - RNNs can be sensitive to LR (gradient explosion/vanishing)
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
        # - RNNs already have implicit regularization through recurrence
        # - Weight decay can interfere with long-term memory in LSTM/GRU
        "weight_decay": {"values": [0]},

        # lr_scheduler: ReduceLROnPlateau configuration
        # - factor=0.5: Halve LR when stuck (standard, well-tested)
        # - patience=8: Wait 8 epochs before reducing
        # - min_lr=1e-6: Floor prevents LR from becoming negligible
        "lr_scheduler_factor": {"values": [0.5]},
        "lr_scheduler_patience": {"values": [8]},
        "lr_scheduler_min_lr": {"values": [1e-6]},

        # gradient_clip_val: Maximum gradient norm
        # CRITICAL for RNNs: Gradients can explode through time (BPTT)
        # - Wider range (0.5-2.0) than transformers
        # - RNNs may need looser clipping to allow gradient flow through sequences
        # - Too tight = starves early timesteps; too loose = instability
        "gradient_clip_val": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 2.0,
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
        #   * Stabilizes RNN training (bounded activations)
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
        # BLOCKRNN ARCHITECTURE
        # ==============================================================================
        # rnn_type: Choice of recurrent cell
        # - LSTM (Long Short-Term Memory):
        #   * Separate cell state for long-term memory
        #   * 4 gates: input, forget, cell, output
        #   * Better at retaining information over long sequences
        #   * May help with conflict patterns spanning years
        # - GRU (Gated Recurrent Unit):
        #   * Combined hidden/cell state (simpler)
        #   * 3 gates: reset, update, new
        #   * Faster training, fewer parameters
        #   * Often comparable performance to LSTM
        "rnn_type": {"values": ["LSTM", "GRU"]},

        # hidden_dim: Hidden state dimension
        # - Controls capacity of temporal memory
        # - 32: Lightweight, fast, less overfitting risk
        # - 64: Balanced capacity for moderate patterns
        # - 128: Higher capacity for complex temporal dynamics
        # - For ~200 series, avoid large dims (256+) to prevent overfitting
        # - Memory scales as O(hidden_dim²) per layer
        "hidden_dim": {"values": [32, 64, 128]},

        # n_rnn_layers: Number of stacked RNN layers
        # - 1 layer: Simple, direct temporal processing
        # - 2 layers: Hierarchical pattern extraction
        # - 3 layers: Deeper abstractions (may have vanishing gradients)
        # - Deeper than 3 rarely helps for time series; increases training time
        "n_rnn_layers": {"values": [1, 2, 3]},

        # activation: Output activation function
        # - ReLU: Fast, sparse activations, standard choice
        # - GELU: Smoother gradients, often slightly better
        # - Tanh removed: Can cause saturation issues in RNNs
        "activation": {"values": ["ReLU", "GELU"]},

        # dropout: Regularization between RNN layers
        # - Applied between stacked layers (not within cells)
        # - LOW values (0.05-0.15) for scarce signal
        # - High dropout would suppress neurons learning rare conflict patterns
        # - RNNs already have implicit regularization through recurrence
        "dropout": {"values": [0.05, 0.15]},

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
        # - False Negative (non-zero→zero): weight = non_zero_weight × fn_weight (8-35)
        #
        # This encourages the model to:
        # 1. Learn strongly from actual conflict events (high non_zero_weight)
        # 2. Penalize missing conflicts (FN penalty)
        # 3. Be forgiving of false alarms (low FP weight encourages exploration)
        "loss_function": {"values": ["WeightedPenaltyHuberLoss"]},

        # zero_threshold: Scaled value below which predictions count as "zero"
        # - After AsinhTransform->MinMaxScaler, 1 fatality ≈ 0.11
        # - Range 0.08-0.23 spans 0-5 fatalities threshold and allows some margin for uncertainty
        # - Lower threshold = stricter zero classification
        "zero_threshold": {
            "distribution": "uniform",
            "min": 0.08, # 0 fatalities threshold after scaling
            "max": 0.23, # 5 fatalities threshold after scaling
        },

        # delta: Huber loss transition point (L2 inside delta, L1 outside)
        # - Range 0.8-1.0 gives nearly pure L2 behavior for [0,1] scaled data
        # - Full L2 maximizes gradient signal from every error
        # - Important for learning from rare spikes where every gradient counts
        "delta": {
            "distribution": "uniform",
            "min": 0.8,
            "max": 1.0,
        },

        # non_zero_weight: Multiplier for non-zero actual values
        # - Fixed at 5.0 to reduce search dimensions
        # - Conflicts contribute 5x more to loss than zeros (counteracts class imbalance)
        # - FP and FN weights are tuned relative to this baseline
        "non_zero_weight": {"values": [5.0]},

        # false_positive_weight: Multiplier when predicting non-zero for actual zero
        # - Range 0.5-1.0 (at or below baseline)
        # - Values <1.0 encourage model to "explore" non-zero predictions
        # - Helps escape local minimum of predicting all zeros
        "false_positive_weight": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1.0,
        },

        # false_negative_weight: Additional multiplier for missing actual conflicts
        # - Applied ON TOP of non_zero_weight: total FN penalty = non_zero × fn_weight
        # - Range 2-8 gives total FN weight of 8-56x baseline
        # - Penalizes missing conflicts (operationally costly)
        # - Narrower range than other models for RNN stability
        "false_negative_weight": {
            "distribution": "uniform",
            "min": 2.0,
            "max": 8.0,
        },
    }

    sweep_config["parameters"] = parameters
    return sweep_config