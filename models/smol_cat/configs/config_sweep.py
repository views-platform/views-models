def get_sweep_config():
    """
    TiDE Hyperparameter Sweep Configuration - MagnitudeAwareQuantileLoss
    =====================================================================
    
    Strategy: Quantile Regression with Magnitude-Aware Weighting
    -------------------------------------------------------------
    Combines asymmetric quantile loss with magnitude-aware weighting to properly
    penalize errors on high-magnitude events in asinh-space.
    
    Why MagnitudeAwareQuantileLoss:
    - Quantile loss: No distributional assumptions → no overflow risk
    - Magnitude scaling: Higher asinh values get proportionally higher weight
    - tau controls asymmetry: tau=0.7 → 2.3× penalty for underestimation
    - Linear magnitude scaling: 1 + |target| → stable gradients
    
    Three-Layer Weighting:
    1. tau (asymmetry): Under-prediction costs tau, over-prediction costs (1-tau)
    2. non_zero_weight: Conflict periods weighted more than zeros
    3. magnitude_mult: 1 + |target| → large events weighted more
    
    Total weight for under-prediction = tau × non_zero_weight × (1 + |target|)
    
    Example (tau=0.7, non_zero_weight=5, target=5.0):
    - weight = 0.7 × 5 × 6 = 21.0
    
    Magnitude Scaling in asinh-space:
    - target = 2.0 (asinh): mult = 3.0x
    - target = 5.0 (asinh): mult = 6.0x  
    - target = 8.0 (asinh): mult = 9.0x
    
    BCD Optimization:
    - Higher tau → catches more events → lower MTD
    - non_zero_weight → focuses on conflict periods → improves MSLE
    - magnitude_mult → large events prioritized → better overall BCD
    """
    sweep_config = {
        "method": "bayes",
        "name": "smol_cat_tide_mag_quantile_v1_msle",
        "early_terminate": {"type": "hyperband", "min_iter": 30, "eta": 2},
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
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
        # Magnitude-aware quantile loss is stable - moderate batch sizes work well
        # Smaller batches = more gradient noise but better rare event detection
        "batch_size": {"values": [64, 128, 256]}, 
        "n_epochs": {"values": [200]},
        "early_stopping_patience": {"values": [30]},
        "early_stopping_min_delta": {"values": [0.0001]},
        "force_reset": {"values": [True]},
        
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        # LR range for batch sizes 64-256 in asinh-transformed space:
        # - Batch 64: use lower end (~2e-5) for stability
        # - Batch 256: can use up to ~1e-4 safely
        # Lower max LR than with batch 512 to prevent instability
        "lr": {
            "distribution": "log_uniform_values",
            "min": 2e-5, 
            "max": 1e-4,
        },
        "weight_decay": {"values": [1e-6]},
        
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["CosineAnnealingWarmRestarts"]},
        "lr_scheduler_T_0": {"values": [25]},
        "lr_scheduler_T_mult": {"values": [1]},
        "lr_scheduler_eta_min": {"values": [1e-6]},
        "gradient_clip_val": {"values": [1.0, 1.5]},
        
        # ==============================================================================
        # SCALING
        # ==============================================================================
        # AsinhTransform compresses heavy-tailed counts into bounded range
        # Loss computed in asinh space (stable), predictions inverse-transformed for metrics
        "feature_scaler": {"values": [None]},
        "target_scaler": {"values": ["AsinhTransform"]},
        
        "feature_scaler_map": {
            "values": [
                {
                    # Conflict history: Asinh transform for heavy tails
                    "AsinhTransform": [
                        "lr_acled_sb", "lr_acled_os",
                        "lr_wdi_sm_pop_refg_or",
                        "lr_wdi_ny_gdp_mktp_kd", "lr_wdi_nv_agr_totl_kn",
                        "lr_splag_1_ged_sb", "lr_splag_1_ged_ns", "lr_splag_1_ged_os",
                    ],
                    # Indices/Rates: Standard scaling
                    "StandardScaler": [
                        "lr_wdi_sm_pop_netm", "lr_wdi_dt_oda_odat_pc_zs",
                        "lr_wdi_sp_pop_grow", "lr_wdi_ms_mil_xpnd_gd_zs",
                        "lr_wdi_sp_dyn_imrt_fe_in", "lr_wdi_sh_sta_stnt_zs",
                        "lr_wdi_sh_sta_maln_zs",
                    ],
                    # Bounded [0,1] features
                    "MinMaxScaler": [
                        "lr_wdi_sl_tlf_totl_fe_zs", "lr_wdi_se_enr_prim_fm_zs",
                        "lr_wdi_sp_urb_totl_in_zs",
                        "lr_vdem_v2x_horacc", "lr_vdem_v2xnp_client", "lr_vdem_v2x_veracc",
                        "lr_vdem_v2x_divparctrl", "lr_vdem_v2xpe_exlpol", "lr_vdem_v2x_diagacc",
                        "lr_vdem_v2xpe_exlgeo", "lr_vdem_v2xpe_exlgender", "lr_vdem_v2xpe_exlsocgr",
                        "lr_vdem_v2x_ex_party", "lr_vdem_v2x_genpp", "lr_vdem_v2xeg_eqdr",
                        "lr_vdem_v2xcl_prpty", "lr_vdem_v2xeg_eqprotec", "lr_vdem_v2x_ex_military",
                        "lr_vdem_v2xcl_dmove", "lr_vdem_v2x_clphy", "lr_vdem_v2xnp_regcorr",
                        # Topics
                        "lr_topic_ste_theta0", "lr_topic_ste_theta1", "lr_topic_ste_theta2",
                        "lr_topic_ste_theta3", "lr_topic_ste_theta4", "lr_topic_ste_theta5",
                        "lr_topic_ste_theta6", "lr_topic_ste_theta7", "lr_topic_ste_theta8",
                        "lr_topic_ste_theta9", "lr_topic_ste_theta10", "lr_topic_ste_theta11",
                        "lr_topic_ste_theta12", "lr_topic_ste_theta13", "lr_topic_ste_theta14",
                        # Topic Lags
                        "lr_topic_ste_theta0_stock_t1_splag", "lr_topic_ste_theta1_stock_t1_splag",
                        "lr_topic_ste_theta2_stock_t1_splag", "lr_topic_ste_theta3_stock_t1_splag",
                        "lr_topic_ste_theta4_stock_t1_splag", "lr_topic_ste_theta5_stock_t1_splag",
                        "lr_topic_ste_theta6_stock_t1_splag", "lr_topic_ste_theta7_stock_t1_splag",
                        "lr_topic_ste_theta8_stock_t1_splag", "lr_topic_ste_theta9_stock_t1_splag",
                        "lr_topic_ste_theta10_stock_t1_splag", "lr_topic_ste_theta11_stock_t1_splag",
                        "lr_topic_ste_theta12_stock_t1_splag", "lr_topic_ste_theta13_stock_t1_splag",
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
        "temporal_width_past": {"values": [12, 24]},
        "temporal_width_future": {"values": [12, 24]},
        "temporal_hidden_size_past": {"values": [128, 256]},
        "temporal_hidden_size_future": {"values": [128, 256]},
        "temporal_decoder_hidden": {"values": [256]},
        
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        "use_layer_norm": {"values": [True, False]},
        "dropout": {"values": [0.1, 0.15, 0.25]},
        "use_static_covariates": {"values": [False, True]},
        "use_reversible_instance_norm": {"values": [True, False]},
        
        # ==============================================================================
        # LOSS FUNCTION: MagnitudeAwareQuantileLoss
        # ==============================================================================
        # Quantile regression with magnitude-aware weighting
        # Combines: tau asymmetry + non_zero_weight + magnitude scaling (1 + |target|)
        # Total weight = tau × non_zero_weight × (1 + |target|)
        "loss_function": {"values": ["MagnitudeAwareQuantileLoss"]},
        
        # tau (quantile level): Controls asymmetry between under/overestimation
        # - tau = 0.5: Symmetric MAE
        # - tau = 0.7: 2.3× penalty for underestimation (FN:FP = 2.3:1)
        # - tau = 0.75: 3× penalty (FN:FP = 3:1)
        # - tau = 0.8: 4× penalty (FN:FP = 4:1)
        # Range 0.60-0.80: Favors catching events without excessive overprediction
        "tau": {
            "distribution": "uniform",
            "min": 0.55,
            "max": 0.80,
        },
        
        # non_zero_weight: Extra weight for samples where target > threshold
        # With ~95% zeros in conflict data, non-zero targets need amplification
        # Combined with magnitude scaling, can use lower values than plain quantile loss
        # Range 1-10: balance with magnitude awareness
        "non_zero_weight": {
            "distribution": "uniform",
            "min": 1.0,
            "max": 10.0,
        },
        
        # zero_threshold: Threshold in asinh-space to distinguish zero from non-zero
        # Also used as reference for magnitude scaling: mult = 1 + |target|/threshold
        # asinh(1)≈0.88, asinh(3)≈1.82, asinh(6)≈2.49, asinh(10)≈3.0
        # Higher threshold = more conservative event detection
        "zero_threshold": {"values": [0.88, 1.82]},
        
        # magnitude_cap: Maximum magnitude multiplier (None = uncapped)
        # With asinh data bounded at ~9, max mult is ~10x which is stable
        # None is recommended for this data
        "magnitude_cap": {"values": [None]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config