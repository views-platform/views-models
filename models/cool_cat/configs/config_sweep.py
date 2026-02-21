def get_sweep_config():
    """
    TiDE Hyperparameter Sweep Configuration - TweedieLoss
    =====================================================

    Strategy: Compound Poisson-Gamma for Zero-Inflated Continuous Data
    ------------------------------------------------------------------
    Uses Tweedie distribution (1 < p < 2) for zero-inflated, right-skewed data.
    Model predicts RAW COUNTS directly (no target transformation).

    Why TweedieLoss:
    - Built-in zero-inflation: Point mass at zero is part of the distribution
    - Proper scoring rule: Theoretically guaranteed to be optimized by the true DGP
    - Scale-aware deviance: Penalizes relative errors more uniformly across scales
    - Compound Poisson-Gamma: Designed for exactly this data shape (90%+ zeros, heavy tail)

    Parameterization:
    - μ (mean): predicted via softplus(model_output), always > 0
    - p (power): controls zero-inflation behavior
      - p closer to 1: More tolerance for zeros (very sparse data)
      - p closer to 2: More weight on positive values (moderate sparsity)
      - p ≈ 1.3: Good starting point for conflict data (~90% zeros)

    Tweedie Deviance:
    D(y, μ) = 2 * [μ^(2-p)/(2-p) - y*μ^(1-p)/(1-p)]  (constant term dropped)

    FN/FP Weighting:
    - Tweedie handles zeros mathematically, so aggressive weighting is less needed
    - false_negative_weight: mild boost (1.0-3.0) to catch missed events
    - false_positive_weight: keep at 1.0 (Tweedie already tolerates zeros)

    LR Considerations:
    - Raw counts → larger gradients than transformed data
    - Use lower LR range: 1e-5 to 2e-4
    - Combined with gradient clipping for stability
    """
    sweep_config = {
        "method": "bayes",
        "name": "cool_cat_tide_tweedie_v10_bcd2",
        "early_terminate": {"type": "hyperband", "min_iter": 30, "eta": 2},
        "metric": {"name": "time_series_wise_bcd_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        "input_chunk_length": {"values": [36, 48]},
        "output_chunk_shift": {"values": [0]},
        "random_state": {"values": [67]},
        "output_chunk_length": {"values": [36]},
        "optimizer_cls": {"values": ["Adam"]},
        "mc_dropout": {"values": [True]},
        "num_samples": {"values": [1]},
        "n_jobs": {"values": [-1]},
        # ==============================================================================
        # TRAINING
        # ==============================================================================
        # With raw counts, larger batches stabilize gradient estimates
        # Batch 64-128 recommended for Tweedie loss stability
        "batch_size": {"values": [64, 128]},
        "n_epochs": {"values": [200]},
        "early_stopping_patience": {"values": [30]},
        "early_stopping_min_delta": {"values": [0.0001]},
        "force_reset": {"values": [True]},
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        # LR range for TweedieLoss with raw counts:
        # - Raw counts → larger gradients → need lower LR
        # - Tweedie deviance is smoother than NB log-gamma
        # - Batch 64-128 with gradient clipping enables slightly higher LR
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 2e-4,
        },
        "weight_decay": {"values": [1e-6]},
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["CosineAnnealingWarmRestarts"]},
        "lr_scheduler_T_0": {"values": [25]},
        "lr_scheduler_T_mult": {"values": [1]},
        "lr_scheduler_eta_min": {"values": [1e-6]},
        # Gradient clipping important for raw count prediction (large gradients)
        "gradient_clip_val": {"values": [1.0, 1.5]},
        # ==============================================================================
        # SCALING
        # ==============================================================================
        # TweedieLoss requires RAW COUNTS (y ≥ 0) for proper Tweedie semantics
        # Model predicts raw counts through softplus (ensures μ > 0)
        # Input features still scaled for stable forward pass
        "feature_scaler": {"values": [None]},
        "target_scaler": {"values": [None]},
        "feature_scaler_map": {
            "values": [
                {
                    # Heavy-tailed features: Asinh transform
                    "AsinhTransform": [
                        "lr_wdi_sm_pop_refg_or",
                        "lr_wdi_ny_gdp_mktp_kd",
                        "lr_wdi_nv_agr_totl_kn",
                        "lr_splag_1_ged_sb",
                        "lr_splag_1_ged_ns",
                        "lr_splag_1_ged_os",
                        # Topic tokens (counts, can be large)
                        "lr_topic_tokens_t1",
                        "lr_topic_tokens_t2",
                        "lr_topic_tokens_t13",
                        "lr_topic_tokens_t1_splag",
                    ],
                    # Can be negative or unbounded: Standard scaling
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
                    # Bounded features: MinMax scaling
                    "MinMaxScaler": [
                        "month",
                        "lr_wdi_sl_tlf_totl_fe_zs",
                        "lr_wdi_se_enr_prim_fm_zs",
                        "lr_wdi_sp_urb_totl_in_zs",
                        # V-Dem (all bounded 0-1)
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
                        # Topic thetas (probabilities 0-1) - temporal lags
                        "lr_topic_ste_theta0_stock_t1",
                        "lr_topic_ste_theta0_stock_t2",
                        "lr_topic_ste_theta0_stock_t13",
                        "lr_topic_ste_theta1_stock_t1",
                        "lr_topic_ste_theta1_stock_t2",
                        "lr_topic_ste_theta1_stock_t13",
                        "lr_topic_ste_theta2_stock_t1",
                        "lr_topic_ste_theta2_stock_t2",
                        "lr_topic_ste_theta2_stock_t13",
                        "lr_topic_ste_theta3_stock_t1",
                        "lr_topic_ste_theta3_stock_t2",
                        "lr_topic_ste_theta3_stock_t13",
                        "lr_topic_ste_theta4_stock_t1",
                        "lr_topic_ste_theta4_stock_t2",
                        "lr_topic_ste_theta4_stock_t13",
                        "lr_topic_ste_theta5_stock_t1",
                        "lr_topic_ste_theta5_stock_t2",
                        "lr_topic_ste_theta5_stock_t13",
                        "lr_topic_ste_theta6_stock_t1",
                        "lr_topic_ste_theta6_stock_t2",
                        "lr_topic_ste_theta6_stock_t13",
                        "lr_topic_ste_theta7_stock_t1",
                        "lr_topic_ste_theta7_stock_t2",
                        "lr_topic_ste_theta7_stock_t13",
                        "lr_topic_ste_theta8_stock_t1",
                        "lr_topic_ste_theta8_stock_t2",
                        "lr_topic_ste_theta8_stock_t13",
                        "lr_topic_ste_theta9_stock_t1",
                        "lr_topic_ste_theta9_stock_t2",
                        "lr_topic_ste_theta9_stock_t13",
                        "lr_topic_ste_theta10_stock_t1",
                        "lr_topic_ste_theta10_stock_t2",
                        "lr_topic_ste_theta10_stock_t13",
                        "lr_topic_ste_theta11_stock_t1",
                        "lr_topic_ste_theta11_stock_t2",
                        "lr_topic_ste_theta11_stock_t13",
                        "lr_topic_ste_theta12_stock_t1",
                        "lr_topic_ste_theta12_stock_t2",
                        "lr_topic_ste_theta12_stock_t13",
                        "lr_topic_ste_theta13_stock_t1",
                        "lr_topic_ste_theta13_stock_t2",
                        "lr_topic_ste_theta13_stock_t13",
                        "lr_topic_ste_theta14_stock_t1",
                        "lr_topic_ste_theta14_stock_t2",
                        "lr_topic_ste_theta14_stock_t13",
                        # Topic thetas - spatial lags (probabilities 0-1)
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
                    ],
                }
            ]
        },
        # ==============================================================================
        # TiDE ARCHITECTURE
        # ==============================================================================
        "num_encoder_layers": {"values": [2]},
        "num_decoder_layers": {"values": [2]},
        "decoder_output_dim": {"values": [128]},
        "hidden_size": {"values": [128, 256, 512]},
        "temporal_width_past": {"values": [36, 48]},
        "temporal_width_future": {"values": [24, 36]},
        "temporal_hidden_size_past": {"values": [128, 256]},
        "temporal_hidden_size_future": {"values": [128, 256]},
        "temporal_decoder_hidden": {"values": [256]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        "use_layer_norm": {"values": [True]},
        "dropout": {"values": [0.15, 0.25]},
        "use_static_covariates": {"values": [True]},
        "use_reversible_instance_norm": {"values": [False]},
        # ==============================================================================
        # LOSS FUNCTION: TweedieLoss
        # ==============================================================================
        # Compound Poisson-Gamma for zero-inflated continuous data
        # Deviance: D(y,μ) ∝ μ^(2-p)/(2-p) - y*μ^(1-p)/(1-p)
        # softplus(pred) → μ > 0
        "loss_function": {"values": ["TweedieLoss"]},
        # p (power parameter): Controls zero-inflation behavior
        # - p closer to 1: More tolerance for zeros (very sparse data)
        # - p closer to 2: More weight on positive values
        # - p ≈ 1.3: Good for conflict data (~90% zeros)
        "p": {
            "distribution": "uniform",
            "min": 1.00,
            "max": 1.25,
        },
        # non_zero_weight: Base weight multiplier for non-zero targets
        # Controls how much more the model prioritizes conflict events
        "non_zero_weight": {
            "distribution": "uniform",
            "min": 1.0,
            "max": 8.0,
        },
        # false_negative_weight: Penalty for missing conflict
        # Tweedie handles zeros mathematically, so mild boost suffices
        "false_negative_weight": {
            "values": [1.0, 2.0, 3.0, 4.0]
        },
        # false_positive_weight: Penalty for false alarms
        # Keep at 1.0 - Tweedie already tolerates zeros naturally
        "false_positive_weight": {
            "values": [1.0, 2.0, 3.0, 4.0]
        },
        # zero_threshold: Raw count threshold to distinguish zero from non-zero
        # For raw counts, 0.5 means < 1 fatality classified as zero
        "zero_threshold": {"values": [0.5]},
        # eps: Numerical stability floor for μ to prevent μ^(-p) explosion
        # 0.01 caps μ^(-1.3) at ~501, preventing gradient spikes in early training
        "eps": {"values": [0.01]},
        # ==============================================================================
        # TEMPORAL ENCODINGS (Position-based)
        # ==============================================================================
        # use_datetime_index: Convert views index to DatetimeIndex
        #   - Required for cyclic encoders (month, week, dayofweek sin/cos)
        #   - NOT required for position encoder (works with integer indices)
        # temporal_precision: Which views index type (month, week, day) - for DatetimeIndex conversion
        #
        # position encoder: relative position in sequence (0.0 to 1.0)
        #   - Works with ANY index type (int or datetime)
        #   - Provides temporal context without requiring DatetimeIndex
        #
        # Note: Cyclic encoding requires DatetimeIndex which has compatibility issues
        # with Darts slicing. Seasonality captured via raw 'month' feature instead.
        # "use_datetime_index": {"values": [False]},
        # "temporal_precision": {"values": ["month"]},  # month | week | day (for future use)
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
