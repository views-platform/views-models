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

    Loss Function: AsymmetricQuantileLoss
    --------------------------------------
    Quantile regression with asymmetric error penalties for zero-inflated data:
    - tau = 0.5: Symmetric (equivalent to MAE)
    - tau = 0.75: 3x penalty for underestimation vs overestimation
    - tau = 0.9: 9x penalty for underestimation vs overestimation
    - non_zero_weight: Additional multiplicative weight for conflict events
    
    Advantages over Huber loss:
    - No delta parameter → avoids MSE-like smoothing that causes flat predictions
    - Natural asymmetry → directly penalizes missing conflicts more than false alarms
    - Linear gradients → better for learning rare spike patterns

    Mode Collapse Prevention:
    -------------------------
    - CosineAnnealingWarmRestarts: periodic LR restarts escape local minima
    - High tau (0.75-0.95): strongly penalizes underestimating conflicts
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
        "name": "cool_cat_tide_aql_v1_bcd2",
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
        "batch_size": {"values": [8, 16, 32, 64]},
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
        # LOSS FUNCTION: AsymmetricQuantileLoss
        # ==============================================================================
        "loss_function": {"values": ["AsymmetricQuantileLoss"]},
        # zero_threshold calibrated to scaled target space [0,1]
        # 0.01-0.05 corresponds to small fatality counts after AsinhTransform->MinMax
        "zero_threshold": {
            "distribution": "uniform",
            "min": 0.01,
            "max": 0.05,
        },
        # tau: quantile level controlling asymmetry
        # tau > 0.5: penalizes underestimation (missing conflicts) more than overestimation
        # tau = 0.75: 3x asymmetry, tau = 0.9: 9x asymmetry, tau = 0.95: 19x asymmetry
        "tau": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 0.95,  # Strong asymmetry (19x) - heavily penalize missing conflicts
        },
        # non_zero_weight: additional multiplicative weight for conflict events
        # Higher values force model to pay more attention to rare conflict samples
        "non_zero_weight": {"values": [1.0, 5.0, 10.0, 15.0, 25.0, 50.0]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config
