def get_sweep_config():
    """
    Transformer Model Hyperparameter Sweep Configuration
    =============================================================================================

    Data Characteristics:
    ---------------------
    - ~200 time series (countries), ~82,512 observations
    - Zero-inflated targets: sb=86%, ns=93%, os=94% zeros
    - Heavy right skew: fatality counts span 0 to ~4,000+
    - 69 features (WDI, V-Dem, topic models, conflict history)
    - 36-month forecast horizon

    Transformer Architecture (Fixed based on literature):
    -----------------------------------------------------
    - Self-attention captures long-range dependencies without recurrence
    - num_encoder_layers=2: Sufficient for ~200 series (Vaswani et al.)
    - num_decoder_layers=2: Match encoder for balanced capacity
    - d_model: 128-256 (moderate size prevents overfitting)
    - num_attention_heads: 4 (head_dim >= 32 required for stability)
    - activation=GEGLU: Gated activation (Shazeer 2020)
    - norm_type=LayerNorm: More stable than RMSNorm
    - use_reversible_instance_norm=True: Critical for distribution shift

    Loss Function: AsinhWeightedPenaltyHuberLoss (Additive Structure)
    -------------------------------------------------------------
    - TN (zero→zero): 1.0x baseline
    - TP (conflict→conflict): 1.0 + non_zero_weight
    - FP (zero→conflict): false_positive_weight (absolute, <0.5 encourages exploration)
    - FN (conflict→zero): 1.0 + non_zero_weight + false_negative_weight

    Mode Collapse Prevention:
    -------------------------
    - CosineAnnealingWarmRestarts: periodic LR restarts escape local minima
    - Low false_positive_weight (<0.5): encourages non-zero predictions
    - Batch size 64-128: ensures non-zero events in every batch
    - Low dropout (0.05-0.1): preserves neurons learning rare patterns

    Attention Stability Constraint:
    --------------------------------
    d_model / num_attention_heads >= 32 required for stable attention.
    Valid combinations: d_model=128/nhead=4 (32✓), d_model=256/nhead=4 (64✓)

    Hyperband Early Termination:
    ----------------------------
    - min_iter=30: More time for sparse signal (was 20)
    - eta=2: Keeps top 50% each round

    Returns:
        sweep_config (dict): WandB sweep configuration dictionary
    """

    sweep_config = {
        "method": "bayes",
        "name": "good_life_transformer_20260214_v1_bcd",
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 30,
            "eta": 2,
        },
        "metric": {"name": "time_series_wise_bcd_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        "input_chunk_length": {"values": [36, 48]},
        "output_chunk_length": {"values": [36]},
        "output_chunk_shift": {"values": [0]},
        "mc_dropout": {"values": [True]},
        "random_state": {"values": [67]},
        "detect_anomaly": {"values": [False]},
        "optimizer_cls": {"values": ["Adam"]},
        "num_samples": {"values": [1]},
        "n_jobs": {"values": [-1]},  # All cores (was 2)

        # ==============================================================================
        # TRAINING BASICS
        # ==============================================================================
        # Batch size 64-128: ~98% probability of non-zero events per batch
        # (was 1024-2048: caused "all-zero batches" → mode collapse)
        "batch_size": {"values": [32, 64]},
        "n_epochs": {"values": [200]},
        "early_stopping_patience": {"values": [30]},  # 40% of T_0 cycle
        "early_stopping_min_delta": {"values": [0.0001]},
        "force_reset": {"values": [True]},

        # ==============================================================================
        # OPTIMIZER / LEARNING RATE SCHEDULE
        # ==============================================================================
        # CosineAnnealing restarts help escape local minima (mode collapse prevention)
        # T_0=50 → 4 cycles in 200 epochs (50, 50, 50, 50)
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-4,  # Raised floor (was 5e-5)
            "max": 2e-3,
        },
        "weight_decay": {"values": [1e-6]},  # Minimal only (was [0, 1e-6, 1e-4, 1e-3])
        "lr_scheduler_cls": {"values": ["CosineAnnealingWarmRestarts"]},
        "lr_scheduler_T_0": {"values": [25]},  # Faster restarts
        "lr_scheduler_T_mult": {"values": [1]},  # Fixed period for sustained exploration
        "lr_scheduler_eta_min": {"values": [1e-6, 1e-5]},  # Higher min maintains gradients
        "gradient_clip_val": {"values": [2.0]},  # Higher clip for larger LR spikes

        # ==============================================================================
        # FEATURE SCALING
        # ==============================================================================
        "feature_scaler": {"values": [None]},
        # AsinhTransform ONLY: preserves zero structure + variance
        # (was AsinhTransform->MinMaxScaler: compressed signal → flat predictions)
        "target_scaler": {"values": ["AsinhTransform"]},
        "feature_scaler_map": {
            "values": [
                {
                    # Zero-inflated counts: Log-like
                    "AsinhTransform": [
                        # "lr_ged_sb", "lr_ged_ns", "lr_ged_os",
                        "lr_acled_sb", "lr_acled_os",
                        "lr_wdi_sm_pop_refg_or",
                        "lr_wdi_ny_gdp_mktp_kd", "lr_wdi_nv_agr_totl_kn",
                        "lr_splag_1_ged_sb", "lr_splag_1_ged_ns", "lr_splag_1_ged_os",
                    ],
                    # Continuous rates/indices: Center around 0 with unit variance
                    "StandardScaler": [
                        "lr_wdi_sm_pop_netm", "lr_wdi_dt_oda_odat_pc_zs",
                        "lr_wdi_sp_pop_grow", "lr_wdi_ms_mil_xpnd_gd_zs",
                        "lr_wdi_sp_dyn_imrt_fe_in", "lr_wdi_sh_sta_stnt_zs",
                        "lr_wdi_sh_sta_maln_zs",
                    ],
                    # V-Dem indices (0-1), WDI %, topic theta
                    "MinMaxScaler": [
                        "lr_wdi_sl_tlf_totl_fe_zs", "lr_wdi_se_enr_prim_fm_zs",
                        "lr_wdi_sp_urb_totl_in_zs",
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
                        "lr_vdem_v2xnp_regcorr",
                        # Topic model theta values (probability distributions, already 0-1)
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
                        # Topic spatial lags (neighborhood averages, still bounded)
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
        # TRANSFORMER ARCHITECTURE
        # ==============================================================================
        # d_model: 128-256 balanced for ~200 series
        # Constraint: d_model / nhead >= 32
        "d_model": {"values": [128, 256]},  # Removed 64 (head_dim too small)

        # num_attention_heads: 4 heads with head_dim >= 32
        # 128/4=32✓, 256/4=64✓
        "num_attention_heads": {"values": [4]},  # FIXED (was [2, 4])

        # num_encoder_layers: 2 sufficient for ~200 series (Vaswani et al.)
        "num_encoder_layers": {"values": [2]},  # FIXED (was [2, 3])

        # num_decoder_layers: Match encoder for balanced capacity
        "num_decoder_layers": {"values": [2]},  # FIXED (was [1, 2])

        # dim_feedforward: 2-4x d_model (standard practice)
        "dim_feedforward": {"values": [256, 512]},

        # dropout: Low to preserve rare pattern learning
        "dropout": {"values": [0.05, 0.1]},  # Reduced from [0.1, 0.2, 0.3]

        # activation: GEGLU (gated activation, Shazeer 2020)
        "activation": {"values": ["GEGLU"]},  # FIXED (was [SwiGLU, GEGLU])

        # norm_type: LayerNorm more stable than RMSNorm
        "norm_type": {"values": ["LayerNorm"]},  # FIXED (was [RMSNorm, LayerNorm])

        # RevIN: Critical for distribution shift in conflict data
        "use_reversible_instance_norm": {"values": [True, False]},

        # ==============================================================================
        # LOSS FUNCTION: AsinhWeightedPenaltyHuberLoss
        # ==============================================================================
        "loss_function": {"values": ["AsinhWeightedPenaltyHuberLoss"]},

        # zero_threshold in ASINH scale (no MinMaxScaler)
        # asinh(1) = 0.88, asinh(25) = 3.91
        "zero_threshold": {
            "distribution": "uniform",
            "min": 0.88,  # ~1 fatality in original scale
            "max": 3.91,  # ~25 fatalities in original scale
        },

        # delta: Huber loss transition (L2 inside, L1 outside)
        # For asinh scale [0, ~9], delta 1-3 is meaningful
        "delta": {
            "distribution": "uniform",
            "min": 1.0,
            "max": 3.0,
        },

        # ==============================================================================
        # LOSS WEIGHTS (Additive structure)
        # ==============================================================================
        # TN = 1.0 (baseline)
        # TP = 1.0 + non_zero_weight
        # FP = false_positive_weight (absolute)
        # FN = 1.0 + non_zero_weight + false_negative_weight

        # non_zero_weight: Focus on the 6-14% signal
        "non_zero_weight": {"values": [5.0, 15.0, 30.0, 50.0]},

        # false_positive_weight: LOW (<0.5) to encourage exploration
        "false_positive_weight": {
            "distribution": "uniform",
            "min": 0.1,
            "max": 0.5,
        },

        # false_negative_weight: Aggressive penalty for missing conflict
        "false_negative_weight": {
            "distribution": "uniform",
            "min": 2.0,
            "max": 50.0,
        },
    }

    sweep_config["parameters"] = parameters
    return sweep_config