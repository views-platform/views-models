def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    Optimized for TransformerModel on country-month level conflict forecasting.
    
    FIXES for weight collapse prevention:
    - Reduced weight_decay to prevent weights shrinking to near-zero
    - Changed target scaler from MinMaxScaler to StandardScaler (preserves gradients)
    - Increased Huber delta for stronger gradient flow
    - Balanced regularization (dropout + weight_decay)
    
    Country-Month Characteristics:
    - ~200 time series (countries) vs ~60k for priogrid
    - Denser, more continuous data per series
    - Can afford larger models due to fewer series
    - Longer temporal patterns (political cycles, economic trends)
    - More stable gradients than sparse priogrid data
    
    Transformer Architecture Notes:
    - Self-attention captures long-range temporal dependencies
    - Multi-head attention allows learning different temporal patterns
    - d_model must be divisible by nhead, and d_model/nhead >= 32 for stability
    - GLU variants (GEGLU, SwiGLU) outperform standard activations
    - RMSNorm is faster than LayerNorm with similar performance
    
    Returns:
    - sweep_config (dict): Configuration for hyperparameter sweeps.
    """

    sweep_config = {
        'method': 'bayes',
        'name': 'good_life_transformer_cm_balanced_v3_mtd',
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 20,  # Higher for scarce signal - transformers need time to find patterns
            'eta': 2
        },
        'metric': {
            'name': 'time_series_wise_mtd_mean_sb',
            'goal': 'minimize'
        },
    }

    parameters = {
        # ============== TEMPORAL CONFIGURATION ==============
        # Country-month has denser data - can use longer lookback windows
        # 48-72 months captures 4-6 year political/economic cycles
        'steps': {'values': [[*range(1, 36 + 1)]]},
        'input_chunk_length': {'values': [48, 60, 72]},  # Longer windows for country-level patterns
        'output_chunk_length': {'values': [36]},
        'output_chunk_shift': {'values': [0]},
        'mc_dropout': {'values': [True]},
        'random_state': {'values': [67]},
        'detect_anomaly': {'values': [False]},

        # ============== TRAINING BASICS ==============
        # Larger batches help stabilize gradients for zero-inflated data
        'batch_size': {'values': [32, 64, 128, 256, 512]},  # Larger for gradient stability
        'n_epochs': {'values': [100]},  # Reduced since we have better hyperparams
        'early_stopping_patience': {'values': [15, 20, 25]},  # More patience for scarce signal
        "early_stopping_min_delta": {"values": [0.00005, 0.0001]},  # Smaller for [0,1] loss scale
        'force_reset': {'values': [True]},

        # ============== OPTIMIZER / SCHEDULER ==============
        # Transformers need careful LR tuning - too high causes instability
        # Rule of thumb: weight_decay should be 10-100x smaller than lr
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 5e-5,
            'max': 1e-3,
        },
        # CRITICAL: Keep weight_decay LOW to prevent weight collapse!
        # High weight_decay caused TiDE weights to shrink to ~1e-34
        # 'weight_decay': {
        #     'distribution': 'log_uniform_values',
        #     'min': 1e-8,
        #     'max': 1e-6,  # Lower ceiling
        # },
        'weight_decay': {'values': [0]},
        'lr_scheduler_factor': {'values': [0.5]},  # Fixed for stability
        'lr_scheduler_patience': {'values': [8]},  # Fixed - consistent plateau detection
        'lr_scheduler_min_lr': {'values': [1e-6]},  # Higher floor to prevent stalling

        "gradient_clip_val": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1.5,
        },

        # ============== SCALING ==============
        # CRITICAL: Use StandardScaler instead of MinMaxScaler to preserve gradients!
        # MinMaxScaler compresses gradients too much, contributing to weight collapse.
        # StandardScaler preserves gradient magnitude while centering/scaling.
        # NOTE: Do NOT use log_targets=True with AsinhTransform - causes double transform and NaN loss!
        'feature_scaler': {'values': [None]},
        # Target is lr_ged_sb - use same scaling as the feature for consistency
        'target_scaler': {'values': ['AsinhTransform->MinMaxScaler']},
        'feature_scaler_map': {
            'values': [{
                # Zero-inflated conflict counts - Asinh + StandardScaler preserves gradients
                "AsinhTransform->MinMaxScaler": [
                    "lr_ged_sb", "lr_ged_ns", "lr_ged_os",
                    "lr_acled_sb", "lr_acled_sb_count", "lr_acled_os",
                    "lr_ged_sb_tsum_24",
                    "lr_splag_1_ged_sb", "lr_splag_1_ged_os", "lr_splag_1_ged_ns",
                    # Large-scale economic data with extreme skew
                    "lr_wdi_ny_gdp_mktp_kd", "lr_wdi_nv_agr_totl_kn",
                    "lr_wdi_sm_pop_netm", "lr_wdi_sm_pop_refg_or",
                ],
                # Bounded percentages, V-Dem indices, and growth rates - StandardScaler works fine
                "MinMaxScaler": [
                    "lr_wdi_sl_tlf_totl_fe_zs", "lr_wdi_se_enr_prim_fm_zs",
                    "lr_wdi_sp_urb_totl_in_zs", "lr_wdi_sh_sta_maln_zs", "lr_wdi_sh_sta_stnt_zs",
                    "lr_wdi_dt_oda_odat_pc_zs", "lr_wdi_ms_mil_xpnd_gd_zs",
                    # V-Dem indices (already 0-1 bounded)
                    "lr_vdem_v2x_horacc", "lr_vdem_v2xnp_client", "lr_vdem_v2x_veracc",
                    "lr_vdem_v2x_divparctrl", "lr_vdem_v2xpe_exlpol", "lr_vdem_v2x_diagacc",
                    "lr_vdem_v2xpe_exlgeo", "lr_vdem_v2xpe_exlgender", "lr_vdem_v2xpe_exlsocgr",
                    "lr_vdem_v2x_ex_party", "lr_vdem_v2x_genpp", "lr_vdem_v2xeg_eqdr",
                    "lr_vdem_v2xcl_prpty", "lr_vdem_v2xeg_eqprotec", "lr_vdem_v2x_ex_military",
                    "lr_vdem_v2xcl_dmove", "lr_vdem_v2x_clphy", "lr_vdem_v2x_hosabort",
                    "lr_vdem_v2xnp_regcorr",
                    # Topic model proportions (0-1 bounded)
                    "lr_topic_ste_theta0", "lr_topic_ste_theta1", "lr_topic_ste_theta2",
                    "lr_topic_ste_theta3", "lr_topic_ste_theta4", "lr_topic_ste_theta5",
                    "lr_topic_ste_theta6", "lr_topic_ste_theta7", "lr_topic_ste_theta8",
                    "lr_topic_ste_theta9", "lr_topic_ste_theta10", "lr_topic_ste_theta11",
                    "lr_topic_ste_theta12", "lr_topic_ste_theta13", "lr_topic_ste_theta14",
                    "lr_topic_ste_theta0_stock_t1_splag", "lr_topic_ste_theta1_stock_t1_splag",
                    "lr_topic_ste_theta2_stock_t1_splag", "lr_topic_ste_theta3_stock_t1_splag",
                    "lr_topic_ste_theta4_stock_t1_splag", "lr_topic_ste_theta5_stock_t1_splag",
                    "lr_topic_ste_theta6_stock_t1_splag", "lr_topic_ste_theta7_stock_t1_splag",
                    "lr_topic_ste_theta8_stock_t1_splag", "lr_topic_ste_theta9_stock_t1_splag",
                    "lr_topic_ste_theta10_stock_t1_splag", "lr_topic_ste_theta11_stock_t1_splag",
                    "lr_topic_ste_theta12_stock_t1_splag", "lr_topic_ste_theta13_stock_t1_splag",
                    "lr_topic_ste_theta14_stock_t1_splag",
                    "lr_topic_tokens_t1", "lr_topic_tokens_t1_splag",
                    # Growth rates (can be negative, roughly normal)
                    "lr_wdi_sp_pop_grow",
                    # Mortality rates (positive, skewed)
                    "lr_wdi_sp_dyn_imrt_fe_in",
                ],
            }]
        },

        # ============== TRANSFORMER ARCHITECTURE ==============
        # Country-month (~200 series) can handle larger models than priogrid
        # d_model/nhead >= 32 required for stable attention
        # Explore range: 128/4=32✓, 256/4=64✓, 256/8=32✓, 512/8=64✓
        'd_model': {'values': [128, 256]},  # Moderate sizes for scarce signal
        'num_attention_heads': {'values': [4, 8]},  # Safe with all d_model values
        
        # Simpler models for scarce signal - avoid overfitting
        'num_encoder_layers': {'values': [1, 2, 3]},
        'num_decoder_layers': {'values': [1, 2]},
        
        # FFN dimension: 2-4x d_model is standard, explore range
        'dim_feedforward': {'values': [512, 1024, 2048]},
        
        # Dropout: LOW for scarce signal - preserve neurons that learn rare patterns
        'dropout': {'values': [0.05, 0.1, 0.15]},
        
        # GLU variants significantly outperform standard activations in Transformers
        # SwiGLU: Best overall (used in LLaMA, PaLM) - smooth gating
        # GEGLU: Strong alternative - GELU-based gating
        # ReGLU: ReLU-based, faster but slightly worse
        'activation': {'values': ['SwiGLU', 'GEGLU', 'gelu']},
        
        # RMSNorm: Faster than LayerNorm, similar performance (used in LLaMA)
        # LayerNormNoBias: Slightly faster, can improve generalization
        'norm_type': {'values': ['RMSNorm', 'LayerNorm']},
        
        # Critical for non-stationary conflict data - True with AsinhTransform
        'use_reversible_instance_norm': {'values': [True, False]},

        # ============== LOSS FUNCTION ==============
        # For zero-inflated data, we need strong gradient flow
        'loss_function': {'values': ['WeightedPenaltyHuberLoss']},
        
        'zero_threshold': {
            'distribution': 'log_uniform_values',
            'min': 0.02,
            'max': 0.1,
        },
        # Delta for Huber loss - tighter range for consistent gradient flow
        'delta': {
            'distribution': 'log_uniform_values',
            'min': 0.1,
            'max': 0.3,
        },
        # Non-zero weight - narrower range for stability
        'non_zero_weight': {
            'distribution': 'uniform',
            'min': 4.0,
            'max': 7.0,   # Narrower range prevents conflicting gradients
        },
        'false_positive_weight': {'values': [1.0]},  # Fixed - variable weighting causes instability
        # Missing conflict is worse than false alarm
        'false_negative_weight': {
            'distribution': 'uniform',
            'min': 2.0,
            'max': 5.0,  # Narrower - still emphasizes missing conflicts
        },
    }

    sweep_config['parameters'] = parameters
    return sweep_config