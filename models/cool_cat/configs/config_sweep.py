def get_sweep_config():
    """
    TiDE Hyperparameter Sweep Configuration - MagnitudeAwareHuberLoss
    ==================================================================
    
    Strategy: Magnitude-Aware Weighted Huber Loss with Linear Scaling
    -----------------------------------------------------------------
    This loss combines the robustness of Huber loss with magnitude-aware
    weighting that scales LINEARLY with target values. Linear scaling provides
    stable gradients while still differentiating between small and large events.
    
    Why MagnitudeAwareHuberLoss:
    - Linear magnitude scaling → stable gradients (vs quadratic which explodes)
    - Huber loss → robust to outliers (L2 for small errors, L1 for large)
    - Additive weight structure → efficient hyperparameter search
    - Magnitude awareness → properly penalizes missing large events in asinh space
    
    Magnitude Scaling (Linear):
    - target = threshold:     mult = 2x
    - target = 2×threshold:   mult = 3x 
    - target = 3×threshold:   mult = 4x
    - target = 4.5×threshold: mult = 5.5x (max for asinh data ~9)
    
    Weight Structure (Additive):
    - TN: 1.0 (base)
    - TP: 1.0 + non_zero_weight
    - FP: false_positive_weight (absolute)
    - FN: 1.0 + non_zero_weight + false_negative_weight
    
    With magnitude scaling:
    - TP_effective = (1 + non_zero_weight) × magnitude_mult
    - FN_effective = (1 + non_zero_weight + fn_weight) × magnitude_mult
    
    BCD Optimization:
    - BCD = ∛(MTD × MSLE × log(1+MSE))
    - false_negative_weight → higher FN penalty → lower MTD
    - non_zero_weight → focuses on conflict periods → improves MSLE
    - AsinhTransform → bounded predictions → controls MSE explosion
    
    Note: AsinhTransform used for bounded training space.
    Loss computed in asinh-space, predictions inverse-transformed for evaluation.
    """
    sweep_config = {
        "method": "bayes",
        "name": "cool_cat_tide_magnitude_v2_bcd",
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
        # MagnitudeAwareHuberLoss with linear scaling is stable but larger batches
        # help smooth gradient estimates. Small batches (64) may need lower LR.
        "batch_size": {"values": [64, 128, 256, 512]}, 
        "n_epochs": {"values": [200]},
        "early_stopping_patience": {"values": [30]},
        "early_stopping_min_delta": {"values": [0.0001]},
        "force_reset": {"values": [True]},
        
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        # Lower LR range for magnitude-aware loss stability
        # Smaller batches (64) should pair with lower LR end (5e-6)
        "lr": {
            "distribution": "log_uniform_values",
            "min": 5e-6, 
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
        # Tighter gradient clipping for magnitude-weighted loss
        "gradient_clip_val": {"values": [0.25, 0.5, 1.0]},
        
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
        "temporal_width_past": {"values": [12]},
        "temporal_width_future": {"values": [12]},
        "temporal_hidden_size_past": {"values": [128, 256]},
        "temporal_hidden_size_future": {"values": [128, 256]},
        "temporal_decoder_hidden": {"values": [256]},
        
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        "use_layer_norm": {"values": [True, False]},
        "dropout": {"values": [0.1, 0.15]},
        "use_static_covariates": {"values": [False, True]},
        "use_reversible_instance_norm": {"values": [True, False]},
        
        # ==============================================================================
        # LOSS FUNCTION: MagnitudeAwareHuberLoss
        # ==============================================================================
        # Magnitude-aware Huber loss with additive weight structure and LINEAR scaling.
        # Linear scaling provides stable gradients while preserving magnitude awareness.
        "loss_function": {"values": ["MagnitudeAwareHuberLoss"]},
        
        # delta: Huber loss L2→L1 transition threshold
        # For asinh-scaled data [0, ~9], delta 1.0-2.0 gives balanced behavior
        # Lower delta = more L1 (robust), higher = more L2 (smooth gradients)
        "delta": {
            "distribution": "uniform",
            "min": 1.0,
            "max": 2.0,
        },
        
        # non_zero_weight: Weight ADDED to base (1.0) for non-zero targets
        # Controls TP weight: TP = 1 + non_zero_weight
        # With ~95% zeros, non-zero targets need amplification
        # Range 3-8: Moderate emphasis, balanced with magnitude scaling
        "non_zero_weight": {
            "distribution": "uniform",
            "min": 3.0,
            "max": 8.0,
        },
        
        # false_positive_weight: ABSOLUTE weight for false positives
        # Values < 1.0 reduce FP penalty, encouraging model to predict events
        # Range 0.3-0.9: Encourages exploration without excessive false alarms
        "false_positive_weight": {
            "distribution": "uniform",
            "min": 0.7,
            "max": 2.0,
        },
        
        # false_negative_weight: Weight ADDED on top of (1 + non_zero_weight) for FN
        # FN = 1 + non_zero_weight + false_negative_weight
        # Controls asymmetry: how much worse is missing a conflict
        # Range 5-20: With linear magnitude scaling (max 5.5x), this gives
        # effective FN weights of ~30-140 at max magnitude
        "false_negative_weight": {
            "distribution": "uniform",
            "min": 2.0,
            "max": 20.0,
        },
        
        # zero_threshold: Threshold in asinh-space to distinguish zero from non-zero
        # Also used as reference for magnitude scaling: mult = 1 + |target|/threshold
        # asinh(1)≈0.88, asinh(3)≈1.82, asinh(6)≈2.49, asinh(10)≈3.0
        # Higher threshold = more conservative event detection
        "zero_threshold": {"values": [1.82, 2.49]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config