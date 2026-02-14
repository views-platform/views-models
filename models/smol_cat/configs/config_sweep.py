def get_sweep_config():
    """
    TiDE Hyperparameter Sweep Configuration for Zero-Inflated Conflict Forecasting
    ================================================================================

    Data Characteristics:
    ---------------------
    - ~200 time series (countries), ~82,512 observations
    - Zero-inflated targets: sb=86%, ns=93%, os=94% zeros
    - Heavy right skew: fatality counts span 0 to ~4,000+
    - 63 features after preprocessing (WDI, V-Dem, topic models, conflict history)
    - 36-month forecast horizon

    Scaling Strategy:
    -----------------
    All features scaled to [0,1] for loss function compatibility:
    - Target: AsinhTransform->MinMaxScaler (handles zeros, bounds output)
    - Conflict counts: AsinhTransform->MinMaxScaler (zero-inflated, extreme skew)
    - Spatial lags: RobustScaler->MinMaxScaler (extreme outliers from aggregation)
    - WDI economics: AsinhTransform->MinMaxScaler (spans orders of magnitude)
    - WDI percentages: StandardScaler->MinMaxScaler (moderate skew, bounded)
    - V-Dem indices: MinMaxScaler (already 0-1 normalized)
    - Topic theta: MinMaxScaler (probability distributions)

    Loss Function: AsinhWeightedPenaltyHuberLoss
    ----------------------------------------
    Asymmetric weighting to handle class imbalance:
    - TN (zero→zero): 1.0x baseline
    - FP (zero→conflict): ~0.1-1.0x (low penalty encourages exploration)
    - TP (conflict→conflict): ~5-15x (high weight for rare events)
    - FN (conflict→zero): ~10-75x (strongest penalty for missing conflicts)

    Mode Collapse Prevention:
    -------------------------
    - CosineAnnealingWarmRestarts: periodic LR restarts escape local minima
    - Low false_positive_weight (<1.0): encourages non-zero predictions
    - Low dropout (0.05-0.15): preserves neurons learning rare patterns
    - Minimal weight_decay (0 or 1e-6): prevents weight collapse

    Architecture (TiDE):
    --------------------
    - Encoder: 2 layers (fixed per TiDE paper)
    - Decoder: 2 layers (fixed per TiDE paper)
    - Hidden size: 128-256 (reduced search space)
    - Temporal width: 12 months (annual cycle - fixed)
    - Layer norm enabled, reversible instance norm always on

    Hyperband Early Termination:
    ----------------------------
    - min_iter=30: time for model to learn from scarce signal
    - eta=2: keeps top 50% each round

    Returns:
        sweep_config (dict): WandB sweep configuration dictionary
    """

    sweep_config = {
        "method": "bayes",
        "name": "smol_cat_tide_20260214_v5_bcd2",
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
        "steps": {"values": [[*range(1, 36 + 1)]]},  # 36-month forecast horizon
        "input_chunk_length": {"values": [36, 48]},  # 3-4 years optimal for 36-month horizon
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
        # Batch size 64+ ensures nearly 100% probability of seeing non-zero events in every batch
        # preventing "dead updates" where model just reinforces zero-prediction
        "batch_size": {"values": [32, 64]},
        "n_epochs": {"values": [200]},
        # patience = 1.2×T_0 to allow one full cycle + buffer after restart
        # With T_0=25, patience=30 ensures model survives at least one restart
        # before early stopping triggers (avoids killing during exploration phase)
        "early_stopping_patience": {"values": [30]},
        "early_stopping_min_delta": {"values": [0.0001]},
        "force_reset": {"values": [True]},
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-4, 
            "max": 8e-3,  # Higher max LR for aggressive restarts
        },
        # Low/zero weight_decay: 1e-4 too aggressive for sparse data
        "weight_decay": {"values": [0, 1e-6]},
        # ==============================================================================
        # LR SCHEDULER: CosineAnnealingWarmRestarts (AGGRESSIVE)
        # ==============================================================================
        # More frequent restarts = more escape attempts from zero-prediction basin
        # T_0=25: 8 cycles in 200 epochs (vs 4 with T_0=50)
        # T_mult=1: constant period for sustained exploration pressure
        # Higher eta_min: maintains gradient flow even at cycle minima
        "lr_scheduler_cls": {"values": ["CosineAnnealingWarmRestarts"]},
        "lr_scheduler_T_0": {"values": [25]},  # Faster restarts
        "lr_scheduler_T_mult": {"values": [1]},  # Fixed period for sustained exploration
        "lr_scheduler_eta_min": {"values": [1e-6, 1e-5]},  # Higher min maintains gradients
        "gradient_clip_val": {"values": [2.0]},  # Higher clip for larger LR spikes
        # ==============================================================================
        # FEATURE SCALING (all bounded to [0,1])
        # ==============================================================================
        "feature_scaler": {"values": [None]},
        "target_scaler": {"values": ["AsinhTransform"]}, # Removed MinMaxScaler to preserve zero-structure and variance
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
        # TiDE ARCHITECTURE
        # ==============================================================================
        "num_encoder_layers": {"values": [2]},
        "num_decoder_layers": {"values": [2]},
        "decoder_output_dim": {"values": [128]},
        "hidden_size": {"values": [128, 256]},
        # Temporal width=12 matches annual cycle in conflict data
        "temporal_width_past": {"values": [12]},  # annual periodicity
        "temporal_width_future": {"values": [12]},  # match past
        "temporal_hidden_size_past": {"values": [128, 256]},  # Reduced search space
        "temporal_hidden_size_future": {"values": [128, 256]},  # Reduced search space
        "temporal_decoder_hidden": {"values": [256]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        "use_layer_norm": {"values": [True, False]},
        "dropout": {"values": [0.05, 0.1]},
        "use_static_covariates": {"values": [False, True]},
        "use_reversible_instance_norm": {"values": [True, False]},
        # ==============================================================================
        # LOSS FUNCTION: AsinhWeightedPenaltyHuberLoss
        # ==============================================================================
        # Asymmetric weighting for zero-inflated data:
        # TN=1x, FP=fp_weight, TP=nz_weight, FN=nz_weight×fn_weight
        "loss_function": {"values": ["AsinhWeightedPenaltyHuberLoss"]},
        # zero_threshold calibrated to scaled target space [0,1]
        # Narrowed to 0.01-0.03 to avoid misclassifying small conflicts as zero
        "zero_threshold": {"values": [1.44]},  # ≈2 fatalities
        # Delta > 1.0 (e.g. 2.0-5.0) focuses on large errors (Quadratic/MSE) while being robust to extreme outliers (Linear)
        # Low delta (0.1) treats everything as outliers (L1), which is bad for noisy data
        "delta": {
            "distribution": "uniform",
            "min": 1.0,
            "max": 3.0,
        },
        # ==============================================================================
        # LOSS WEIGHTS (Magnitude-aware: mult = 1 + (target/threshold)²)
        # ==============================================================================
        # Lower base weights - magnitude scaling handles large events automatically
        # mult ranges from 2× (small events) to 40× (large events)
        "non_zero_weight": {"values": [10.0, 20.0, 30.0]},
        
        # false_positive_weight: Balanced range for exploration
        "false_positive_weight": {
            "distribution": "uniform",
            "min": 0.3,
            "max": 0.7,
        },
        
        # false_negative_weight: Additional penalty for missing conflicts
        # Combined with non_zero_weight for total FN weight
        "false_negative_weight": {
            "distribution": "uniform",
            "min": 5.0,
            "max": 20.0,
        },
    }

    sweep_config["parameters"] = parameters
    return sweep_config
