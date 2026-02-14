def get_sweep_config():
    """
    TCN Hyperparameter Sweep Configuration for Zero-Inflated Conflict Forecasting
    ================================================================================

    Data Characteristics:
    ---------------------
    - ~200 time series (countries), ~82,512 observations
    - Zero-inflated targets: sb=86%, ns=93%, os=94% zeros
    - Heavy right skew: fatality counts span 0 to ~4,000+
    - 63 features after preprocessing (WDI, V-Dem, topic models, conflict history)
    - 36-month forecast horizon

    TCN Architecture Rationale (Literature-Informed):
    --------------------------------------------------
    TCNs use causal dilated convolutions to capture long-range dependencies efficiently.
    
    Receptive field = 1 + (kernel_size - 1) × Σ(dilation_base^i) for i ∈ [0, num_layers)
    
    For our 36-month horizon with sparse events, we design:
    - kernel_size=3: Small kernels better for sparse signals (Bai et al., 2018)
    - dilation_base=2: Standard exponential growth
    - num_layers=6: RF = 1 + 2×(1+2+4+8+16+32) = 127 months ✓
    
    Alternative config (kernel_size=5, num_layers=4):
    - RF = 1 + 4×(1+2+4+8) = 61 months ✓ Also sufficient
    
    For zero-inflated data:
    - Use weight_norm (not batch_norm) - more stable with sparse gradients
    - Low dropout (0.05-0.15) - preserve neurons learning rare patterns
    - Moderate num_filters (64-128) - avoid overfitting to dominant zeros

    Scaling Strategy:
    -----------------
    Target: AsinhTransform ONLY (no MinMaxScaler chain!)
    - Preserves zero structure for loss function classification
    - Maintains variance signal for gradient flow
    - Outputs in [0, ~9] for typical conflict data

    Loss Function: AsinhWeightedPenaltyHuberLoss
    ---------------------------------------------
    Magnitude-aware asymmetric weighting:
    - TN: 1.0× baseline
    - TP: (1 + non_zero_weight) × magnitude_mult
    - FP: false_positive_weight (absolute, <0.5 encourages exploration)
    - FN: (1 + non_zero_weight + false_negative_weight) × magnitude_mult
    
    Key insight: non_zero_weight ≥ 30 prevents gradient collapse
    (model stays engaged with conflict events even when predicting well)

    Mode Collapse Prevention:
    -------------------------
    - CosineAnnealingWarmRestarts: T_0=25-30 for frequent escapes from zero-prediction basin
    - Small batch_size (32-64): ensures ~5+ non-zero samples per batch
    - High non_zero_weight: maintains gradient signal from rare events
    - Low false_positive_weight: reduces penalty for predicting conflict

    Hyperband Early Termination:
    ----------------------------
    - min_iter=30: allows full learning cycle before termination
    - eta=2: keeps top 50% each round

    Returns:
        sweep_config (dict): WandB sweep configuration dictionary
    """

    sweep_config = {
        "method": "bayes",
        "name": "teenage_dirtbag_tcn_20260214_bcd",
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 30,  # Allow full restart cycle
            "eta": 2,
        },
        "metric": {"name": "time_series_wise_bcd_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        "input_chunk_length": {"values": [36, 48]},  # Reduced: longer doesn't help for sparse data
        "output_chunk_shift": {"values": [0]},
        "output_chunk_length": {"values": [36]},
        "random_state": {"values": [67]},
        "n_jobs": {"values": [-1]},
        
        # ==============================================================================
        # TRAINING BASICS
        # ==============================================================================
        # Small batch size critical for zero-inflated data:
        # 32 samples × 15% non-zero = ~5 conflict events per batch
        # Ensures non-zero events in EVERY gradient update
        "batch_size": {"values": [32, 64]},
        "n_epochs": {"values": [200]},
        # patience > T_0 to survive LR restart dips
        "early_stopping_patience": {"values": [30]},
        "early_stopping_min_delta": {"values": [0.0001]},
        "force_reset": {"values": [True]},
        "save_checkpoints": {"values": [False]},
        
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        "optimizer_cls": {"values": ["Adam"]},
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 5e-3,  # TCNs can handle moderate LR
        },
        # Zero/minimal weight_decay - prevents weight collapse on sparse data
        "weight_decay": {"values": [0, 1e-6]},
        
        # ==============================================================================
        # LR SCHEDULER: CosineAnnealingWarmRestarts (NOT ReduceLROnPlateau!)
        # ==============================================================================
        # ReduceLROnPlateau is BAD for zero-inflated data:
        # - When model collapses to zeros, loss plateaus
        # - RLROP reduces LR, making escape HARDER
        # 
        # CosineAnnealingWarmRestarts:
        # - Periodic LR spikes help escape zero-prediction basin
        # - T_0=25: 8 restart cycles in 200 epochs
        "lr_scheduler_cls": {"values": ["CosineAnnealingWarmRestarts"]},
        "lr_scheduler_T_0": {"values": [25]},
        "lr_scheduler_T_mult": {"values": [1]},  # Fixed period for sustained exploration
        "lr_scheduler_eta_min": {"values": [1e-6, 1e-5]},
        "gradient_clip_val": {"values": [ 2.0]},  # Allow strong FN gradients
        
        # ==============================================================================
        # SCALING
        # ==============================================================================
        # CRITICAL: AsinhTransform ONLY for targets!
        # MinMaxScaler chains compress the variance signal → flat predictions
        "feature_scaler": {"values": [None]},
        "target_scaler": {"values": ["AsinhTransform"]},
        "feature_scaler_map": {
            "values": [
                {
                    # Zero-inflated conflict counts (require asinh for zeros + extreme skew)
                    "AsinhTransform": [
                        "lr_acled_sb", "lr_acled_os",
                        "lr_wdi_sm_pop_refg_or",
                        "lr_wdi_ny_gdp_mktp_kd", "lr_wdi_nv_agr_totl_kn",
                        "lr_splag_1_ged_sb", "lr_splag_1_ged_ns", "lr_splag_1_ged_os",
                    ],
                    # Rates with negative values (growth rates, net migration)
                    "StandardScaler": [
                        "lr_wdi_sm_pop_netm", "lr_wdi_dt_oda_odat_pc_zs",
                        "lr_wdi_sp_pop_grow", "lr_wdi_ms_mil_xpnd_gd_zs",
                        "lr_wdi_sp_dyn_imrt_fe_in", "lr_wdi_sh_sta_stnt_zs",
                        "lr_wdi_sh_sta_maln_zs",
                    ],
                    # Bounded variables (V-Dem 0-1, percentages, topic proportions)
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
                        # Topic proportions (0-1 bounded)
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
                    ],
                }
            ]
        },
        
        # ==============================================================================
        # TCN ARCHITECTURE (Literature-Optimized for Sparse Data)
        # ==============================================================================
        # kernel_size: Small kernels (3) work better for sparse signals
        # - Each kernel "sees" fewer timesteps, reducing zero-dilution
        # - kernel_size=3 sufficient for most sequence tasks
        "kernel_size": {"values": [3]},  # Fixed: 3 is optimal for sparse data
        
        # num_filters: Moderate capacity to avoid overfitting to zeros
        # - Too many filters → learn to predict zeros everywhere
        # - 64-128 provides enough capacity without overfitting
        "num_filters": {"values": [64, 128]},
        
        # dilation_base: Standard exponential growth
        # - 2 is canonical (powers of 2: 1,2,4,8,16,32...)
        "dilation_base": {"values": [2]},  # Fixed: standard choice
        
        # num_layers: Controls receptive field
        # RF = 1 + (k-1) × Σ(d^i) = 1 + 2 × (1+2+4+8+16+32) = 127 months with 6 layers
        # None = auto-compute to cover input_chunk_length
        "num_layers": {"values": [5, 6]},  # 5: RF=63, 6: RF=127
        
        # dropout: LOW for sparse data - preserve neurons learning rare patterns
        "dropout": {"values": [0.05, 0.1]},
        
        # weight_norm: Essential for TCN stability (better than batch_norm for sparse data)
        "weight_norm": {"values": [True]},
        
        # Reversible instance norm: Helps with distribution shift in time series
        "use_reversible_instance_norm": {"values": [True]},
        
        # ==============================================================================
        # LOSS FUNCTION: AsinhWeightedPenaltyHuberLoss
        # ==============================================================================
        "loss_function": {"values": ["AsinhWeightedPenaltyHuberLoss"]},
        
        # zero_threshold: In ASINH scale (not raw counts)
        # asinh(1) ≈ 0.88, asinh(25) ≈ 3.91
        "zero_threshold": {
            "distribution": "uniform",
            "min": 0.88,
            "max": 3.91,
        },
        
        # delta: Huber L2→L1 transition point
        # For asinh data in [0, ~9], delta 1-3 gives balanced behavior
        "delta": {
            "distribution": "uniform",
            "min": 1.0,
            "max": 3.0,
        },
        
        # non_zero_weight: - prevents gradient collapse!
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