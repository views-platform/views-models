def get_sweep_config():
    """
    TSMixer Hyperparameter Sweep Configuration for Zero-Inflated Conflict Forecasting
    ==================================================================================
    
    Data Characteristics:
    ---------------------
    - ~200 time series (countries), ~82,512 observations
    - Zero-inflated targets: sb=86%, ns=93%, os=94% zeros
    - Heavy right skew: fatality counts span 0 to ~4,000+
    - 69 features after preprocessing (WDI, V-Dem, topic models, conflict history)
    - 36-month forecast horizon

    TSMixer Architecture (Google 2023):
    -----------------------------------
    - MLP-based mixer: alternates time-mixing and feature-mixing layers
    - Simpler than Transformers, fewer parameters, faster training
    - num_blocks=2: Sufficient for ~200 series (prevents overfitting)
    - ff_size = 2×hidden_size: Standard expansion ratio
    - normalize_before=True: Pre-norm more stable (Transformer literature)
    - activation=GELU: Smoother than ReLU (TSMixer paper default)
    - use_reversible_instance_norm=True: Critical for distribution shift

    Loss Function: AsinhWeightedPenaltyHuberLoss (Additive Structure)
    -------------------------------------------------------------
    - TN (zero→zero): 1.0x baseline
    - TP (conflict→conflict): 1.0 + non_zero_weight
    - FP (zero→conflict): false_positive_weight (absolute, <1.0 encourages exploration)
    - FN (conflict→zero): 1.0 + non_zero_weight + false_negative_weight

    Mode Collapse Prevention:
    -------------------------
    - CosineAnnealingWarmRestarts: periodic LR restarts escape local minima
    - Low false_positive_weight (<0.5): encourages non-zero predictions
    - Batch size 64-128: ensures non-zero events in every batch
    - Low dropout (0.05-0.1): preserves neurons learning rare patterns

    Hyperband Early Termination:
    ----------------------------
    - min_iter=30: time for model to learn from scarce signal
    - eta=2: keeps top 50% each round

    Returns:
        sweep_config (dict): WandB sweep configuration dictionary
    """

    sweep_config = {
        "method": "bayes",
        "name": "elastic_heart_tsmixer_20260214_v1_bcd",
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 30,  # More time for sparse signal (was 20)
            "eta": 2,
        },
        "metric": {
            "name": "time_series_wise_bcd_mean_sb",
            "goal": "minimize",
        },
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        "input_chunk_length": {"values": [36, 48]},  # Reduced from [36,48,60] - 48 sufficient
        "output_chunk_length": {"values": [36]},
        "output_chunk_shift": {"values": [0]},
        "random_state": {"values": [67]},
        "mc_dropout": {"values": [True]},
        "optimizer_cls": {"values": ["Adam"]},
        "num_samples": {"values": [1]},
        "n_jobs": {"values": [-1]},

        # ==============================================================================
        # TRAINING 
        # ==============================================================================
        "batch_size": {"values": [32, 64]},
        "n_epochs": {"values": [200]},  # Increased from 150 for CosineAnnealing
        "early_stopping_patience": {"values": [30]},  # 40% of T_0 cycle
        "early_stopping_min_delta": {"values": [0.0001]},
        "force_reset": {"values": [True]},

        # ==============================================================================
        # OPTIMIZER: CosineAnnealingWarmRestarts (replaces ReduceLROnPlateau)
        # ==============================================================================
        # CosineAnnealing restarts help escape local minima (mode collapse prevention)
        # T_0=50 → 4 cycles in 200 epochs (50, 50, 50, 50)
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-4,  # Raised floor (was 5e-5)
            "max": 5e-3,  # Raised ceiling for larger batches
        },
        "weight_decay": {"values": [1e-6]},  # Minimal (was 0 - can cause instability)
        "lr_scheduler_cls": {"values": ["CosineAnnealingWarmRestarts"]},
        "lr_scheduler_T_0": {"values": [25]},  # Faster restarts
        "lr_scheduler_T_mult": {"values": [1]},  # Fixed period for sustained exploration
        "lr_scheduler_eta_min": {"values": [1e-6, 1e-5]},  # Higher min maintains gradients
        "gradient_clip_val": {"values": [2.0]},  # Higher clip for larger LR spikes

        # ==============================================================================
        # SCALING (No MinMaxScaler chains - preserves variance)
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
        # TSMIXER ARCHITECTURE (Fixed based on literature)
        # ==============================================================================
        # Google TSMixer (2023): 2 blocks sufficient for small datasets
        # More blocks → overfitting on ~200 time series
        "num_blocks": {"values": [2]},  # FIXED (was [2,3,4])

        # hidden_size: Main representation dimension
        # 128-256 reasonable for 69 features
        "hidden_size": {"values": [128, 256]},

        # ff_size: Feed-forward expansion = 2×hidden_size (TSMixer paper default)
        # FIXED relationship reduces search space
        "ff_size": {"values": [256, 512]},  # Matches 2×[128, 256]

        # normalize_before=True: Pre-normalization more stable (Transformer literature)
        "normalize_before": {"values": [True, False]},  # FIXED (was [True, False])

        # activation=GELU: Smoother gradients than ReLU (TSMixer paper default)
        "activation": {"values": ["GELU"]},  # FIXED (was [ReLU, GELU])

        # LayerNorm: Standard for MLP-mixers
        "norm_type": {"values": ["LayerNorm"]},

        # Dropout: Low to preserve rare pattern learning
        "dropout": {"values": [0.05, 0.1]},

        # RevIN: Critical for distribution shift in conflict data
        "use_reversible_instance_norm": {"values": [True, False]},  # FIXED (was [True, False])

        # No static covariates in this dataset
        "use_static_covariates": {"values": [False, True]},  # FIXED (was True)

        # ==============================================================================
        # LOSS FUNCTION: AsinhWeightedPenaltyHuberLoss (Additive structure)
        # ==============================================================================
        "loss_function": {"values": ["AsinhWeightedPenaltyHuberLoss"]},

        # zero_threshold in ASINH scale (no MinMaxScaler)
        # asinh(1) = 0.88, asinh(25) = 3.91
        # Threshold for "is this conflict or peace?"
        "zero_threshold": {"values": [1.44]},  # ≈2 fatalities

        # delta: Huber loss transition (L2 inside, L1 outside)
        # For asinh scale [0, ~9], delta 1-3 is meaningful
        # delta=1.2: L2 for small conflicts, L1 for large wars
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

        # Values ≥30 keep model engaged with conflict events
        "non_zero_weight": {"values": [30.0, 50.0, 75.0]},
        
        # false_positive_weight: Low values encourage exploration
        # < 0.5 means FP is cheaper than TN, pushing model to predict conflicts
        "false_positive_weight": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1.0,
        },
        
        # false_negative_weight: Additional penalty for missing conflicts
        # Combined with non_zero_weight for total FN weight
        "false_negative_weight": {
            "distribution": "uniform",
            "min": 5.0,
            "max": 30.0,
        },
    }

    sweep_config["parameters"] = parameters
    return sweep_config