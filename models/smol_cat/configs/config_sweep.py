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

    Loss Function: WeightedPenaltyHuberLoss
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
    - Encoder: 2-3 layers, hidden_size 64-256
    - Decoder: 1-2 layers, output_dim 64-128
    - Temporal width: 6-18 months (12 captures annual cycles)
    - Layer norm enabled, reversible instance norm explored

    Hyperband Early Termination:
    ----------------------------
    - min_iter=30: time for model to learn from scarce signal
    - eta=2: keeps top 50% each round

    Returns:
        sweep_config (dict): WandB sweep configuration dictionary
    """

    sweep_config = {
        "method": "bayes",
        "name": "smol_cat_tide_end_me_bcd2",
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
        "input_chunk_length": {"values": [24, 36, 48, 72]},  # 2-6 years of context
        "output_chunk_shift": {"values": [0]},
        "mc_dropout": {"values": [True]},
        "random_state": {"values": [67]},
        "output_chunk_length": {"values": [36]},
        "optimizer_cls": {"values": ["Adam"]},
        "num_samples": {"values": [1]},
        "n_jobs": {"values": [2]},
        # ==============================================================================
        # TRAINING
        # ==============================================================================
        "batch_size": {"values": [1024, 2048]},  # Large batches for zero-inflated data
        "n_epochs": {"values": [200]},
        # patience > max(T_0) to allow recovery after LR restart
        "early_stopping_patience": {"values": [35]},
        "early_stopping_min_delta": {"values": [0.0001]},
        "force_reset": {"values": [True]},
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        "lr": {
            "distribution": "log_uniform_values",
            "min": 5e-5,
            "max": 1e-3,
        },
        # Low/zero weight_decay: high values caused weight collapse in prior runs
        "weight_decay": {"values": [0, 1e-6, 1e-4]},
        # ==============================================================================
        # LR SCHEDULER: CosineAnnealingWarmRestarts
        # ==============================================================================
        # Periodic restarts help escape local minima
        "lr_scheduler_cls": {"values": ["CosineAnnealingWarmRestarts"]},
        "lr_scheduler_T_0": {"values": [20, 30, 40]},  # T_0 ≥ patience/2. 
        "lr_scheduler_T_mult": {"values": [1, 2]},  # 1=fixed period, 2=double each restart, can cause early stopping to trigger prematurely right after a restart
        "lr_scheduler_eta_min": {"values": [1e-6, 1e-5]},
        "gradient_clip_val": {"values": [1.5]},
        # ==============================================================================
        # FEATURE SCALING (all bounded to [0,1])
        # ==============================================================================
        "feature_scaler": {"values": [None]},
        "target_scaler": {"values": ["AsinhTransform->MinMaxScaler"]},
        "feature_scaler_map": {
            "values": [
                {
                    # Zero-inflated counts and multiplicative data
                    "AsinhTransform->MinMaxScaler": [
                        "lr_ged_sb", "lr_ged_ns", "lr_ged_os",
                        "lr_acled_sb", "lr_acled_os",
                        "lr_wdi_sm_pop_refg_or",
                        "lr_wdi_ny_gdp_mktp_kd", "lr_wdi_nv_agr_totl_kn",
                    ],
                    # Spatial lags with extreme outliers
                    "RobustScaler->MinMaxScaler": [
                        "lr_splag_1_ged_sb", "lr_splag_1_ged_ns", "lr_splag_1_ged_os",
                    ],
                    # WDI indicators with moderate skew or negatives
                    "StandardScaler->MinMaxScaler": [
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
        "num_encoder_layers": {"values": [2, 3]},
        "num_decoder_layers": {"values": [1, 2]},
        "decoder_output_dim": {"values": [64, 128]},
        "hidden_size": {"values": [64, 128, 256]},
        # Temporal widths include 12 for annual cycle detection
        "temporal_width_past": {"values": [6, 12, 18]},
        "temporal_width_future": {"values": [6, 12, 18]},
        "temporal_hidden_size_past": {"values": [64, 128, 256]},
        "temporal_hidden_size_future": {"values": [64, 128, 256]},
        "temporal_decoder_hidden": {"values": [128, 256]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        "use_layer_norm": {"values": [True]},
        "dropout": {"values": [0.05, 0.1, 0.15]},  # Low to preserve rare patterns
        "use_static_covariates": {"values": [False]},
        "use_reversible_instance_norm": {"values": [True, False]},
        # ==============================================================================
        # LOSS FUNCTION: WeightedPenaltyHuberLoss
        # ==============================================================================
        # Asymmetric weighting for zero-inflated data:
        # TN=1x, FP=fp_weight, TP=nz_weight, FN=nz_weight×fn_weight
        "loss_function": {"values": ["WeightedPenaltyHuberLoss"]},
        # zero_threshold calibrated to scaled target space [0,1]
        # 0.03-0.08 corresponds to ~1-3 fatalities after AsinhTransform->MinMax. Alternative: 0.019 - 0.133 
        "zero_threshold": {
            "distribution": "uniform",
            "min": 0.019,
            "max": 0.133,
        },
        # High delta gives near-L2 behavior (maximizes gradient signal)
        "delta": {
            "distribution": "uniform",
            "min": 0.4,
            "max": 1.0,
        },
        # ==============================================================================
        # LOSS WEIGHTS (Mode Collapse Prevention)
        # ==============================================================================
        # Key ratios: FN:FP should be 40-100x to discourage missing conflicts
        # fp_weight < 1.0 encourages model to explore non-zero predictions
        "non_zero_weight": {"values": [5.0, 10.0, 15.0]},
        "false_positive_weight": {
            "distribution": "uniform",
            "min": 0.1,
            "max": 1.0,
        },
        "false_negative_weight": {
            "distribution": "uniform",
            "min": 2.0,
            "max": 10.0,
        },
    }

    sweep_config["parameters"] = parameters
    return sweep_config
