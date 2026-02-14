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
        "name": "cool_cat_tide_aql_v2_bcd2",
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
        "mc_dropout": {"values": [True]},
        "random_state": {"values": [67]},
        "output_chunk_length": {"values": [36]},
        "optimizer_cls": {"values": ["Adam"]},
        "num_samples": {"values": [1]},
        "n_jobs": {"values": [2]},
        # ==============================================================================
        # TRAINING
        # ==============================================================================
        "batch_size": {"values": [8, 16, 32]},  # Reduced: 64 dilutes spike signal too much
        "n_epochs": {"values": [200]},
        # patience = 1.3×T_0 to allow full cycle after restart
        "early_stopping_patience": {"values": [40]},
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
        # Low/zero weight_decay: 1e-4 too aggressive for sparse data
        "weight_decay": {"values": [0, 1e-6]},
        # ==============================================================================
        # LR SCHEDULER: CosineAnnealingWarmRestarts
        # ==============================================================================
        # Periodic restarts help escape local minima
        "lr_scheduler_cls": {"values": ["CosineAnnealingWarmRestarts"]},
        "lr_scheduler_T_0": {"values": [30]},  # Optimal for 200 epochs with patience=40
        "lr_scheduler_T_mult": {"values": [1]},  # Fixed period for sparse data
        "lr_scheduler_eta_min": {"values": [1e-6]},
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
        "num_encoder_layers": {"values": [2]},  # Fixed: optimal per TiDE paper ablations
        "num_decoder_layers": {"values": [2]},  # Fixed: optimal per TiDE paper ablations
        "decoder_output_dim": {"values": [128]},  # Fixed: medium capacity sufficient
        "hidden_size": {"values": [128, 256]},  # Reduced: 64 too small, focus on key values
        # Temporal width=12 matches annual cycle in conflict data
        "temporal_width_past": {"values": [12]},  # Fixed: annual periodicity
        "temporal_width_future": {"values": [12]},  # Fixed: match past
        "temporal_hidden_size_past": {"values": [128, 256]},  # Reduced search space
        "temporal_hidden_size_future": {"values": [128, 256]},  # Reduced search space
        "temporal_decoder_hidden": {"values": [256]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        "use_layer_norm": {"values": [True, False]},
        "dropout": {"values": [0.05, 0.1]},
        "use_static_covariates": {"values": [False]},
        "use_reversible_instance_norm": {"values": [True, False]},
        # ==============================================================================
        # LOSS FUNCTION: AsymmetricQuantileLoss
        # ==============================================================================
        "loss_function": {"values": ["AsymmetricQuantileLoss"]},
        # zero_threshold calibrated to scaled target space [0,1]
        # Narrowed to 0.01-0.03 to avoid misclassifying small conflicts as zero
        "zero_threshold": {
            "distribution": "uniform",
            "min": 0.01,
            "max": 0.03,
        },
        # tau: quantile level controlling asymmetry
        # tau > 0.5: penalizes underestimation (missing conflicts) more
        # Narrowed to 0.7-0.95 to focus on strong asymmetry for conflict detection
        "tau": {
            "distribution": "uniform",
            "min": 0.7,   # Moderate asymmetry (2.33x)
            "max": 0.95,  # Strong asymmetry (19x)
        },
        # non_zero_weight: additional multiplicative weight for conflict events
        # Reduced to key values for efficiency
        "non_zero_weight": {"values": [1.0, 10.0, 25.0, 50.0]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config
