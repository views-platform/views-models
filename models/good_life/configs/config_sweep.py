def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    Optimized for TransformerModel on country-month level conflict forecasting.
    
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
        'name': 'good_life_transformer_cm_balanced_v1_mtd',
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 15,  # Slightly higher - transformers need more warmup
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

        # ============== TRAINING BASICS ==============
        # Smaller batches work better with ~200 countries (more gradient noise = regularization)
        'batch_size': {'values': [16, 32, 64]},  # Smaller for country-month
        'n_epochs': {'values': [300]},
        'early_stopping_patience': {'values': [6]},  # More patience for stable convergence
        'early_stopping_min_delta': {'values': [0.0005, 0.001]},  # Tighter threshold
        'force_reset': {'values': [True]},

        # ============== OPTIMIZER / SCHEDULER ==============
        # Country-month is more stable - can use slightly higher LR than priogrid
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 5e-6,
            'max': 1e-4,  # Slightly higher max since denser data
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-3,
        },
        'lr_scheduler_factor': {
            'distribution': 'uniform',
            'min': 0.3,
            'max': 0.5,
        },
        'lr_scheduler_patience': {'values': [3]},  # More patience
        'lr_scheduler_min_lr': {'values': [1e-7]},
        # Can relax gradient clipping slightly for denser country data
        'gradient_clip_val': {
            'distribution': 'uniform',
            'min': 0.2,
            'max': 1.0,
        },

        # ============== SCALING ==============
        # RobustScaler as default fallback for unmapped features
        # feature_scaler_map assigns optimal scalers based on feature characteristics:
        # - AsinhTransform: Zero-inflated counts (fatalities) and heavily skewed economic data
        # - MinMaxScaler: Bounded percentages (0-100), V-Dem indices (0-1), topic proportions
        # - StandardScaler: Growth rates (normal-ish, can be negative)
        # - SqrtTransform: Mortality rates (positive, moderate skew)
        # NOTE: Do NOT use log_targets=True with AsinhTransform - causes double transform and NaN loss!
        'feature_scaler': {'values': [None]},
        'target_scaler': {'values': ['AsinhTransform->MinMaxScaler']},  # Fixed best option
        'feature_scaler_map': {
            'values': [{
                # Zero-inflated conflict counts - asinh handles zeros and extreme spikes
                "AsinhTransform->MinMaxScaler": [
                    "lr_ged_sb", "lr_ged_ns", "lr_ged_os",
                    "lr_acled_sb", "lr_acled_sb_count", "lr_acled_os",
                    "lr_ged_sb_tsum_24",
                    "lr_splag_1_ged_sb", "lr_splag_1_ged_os", "lr_splag_1_ged_ns",
                    # Large-scale economic data with extreme skew
                    "lr_wdi_ny_gdp_mktp_kd", "lr_wdi_nv_agr_totl_kn",
                    "lr_wdi_sm_pop_netm", "lr_wdi_sm_pop_refg_or"
                ],
                # Bounded percentages and rates (0-100 scale)
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
                    "lr_topic_ste_theta14_stock_t1_splag"
                ],
                # Growth rates (can be negative, roughly normal)
                "StandardScaler->MinMaxScaler": [
                    "lr_wdi_sp_pop_grow"
                ],
                # Mortality rates (positive, moderate skew)
                "SqrtTransform->MinMaxScaler": [
                    "lr_wdi_sp_dyn_imrt_fe_in"
                ],
                # Token counts (moderate skew)
                "RobustScaler->MinMaxScaler": [
                    "lr_topic_tokens_t1", "lr_topic_tokens_t1_splag"
                ]
            }]
        },

        # ============== TRANSFORMER ARCHITECTURE ==============
        # Country-month (~200 series) can handle larger models than priogrid
        # d_model/nhead >= 32 required for stable attention
        # Explore range: 128/4=32✓, 256/4=64✓, 256/8=32✓, 512/8=64✓
        'd_model': {'values': [128, 256, 512]},  # Explore wider range for country-month
        'num_attention_heads': {'values': [4, 8]},  # Safe with all d_model values
        
        # Deeper models OK for denser country data (2-4 layers)
        'num_encoder_layers': {'values': [2, 3, 4]},  # Can go deeper with country-month
        'num_decoder_layers': {'values': [2, 3]},  # Decoder can be shallower
        
        # FFN dimension: 2-4x d_model is standard, explore range
        'dim_feedforward': {'values': [512, 1024, 2048]},
        
        # Dropout: slightly higher for regularization with small dataset
        'dropout': {'values': [0.1, 0.15, 0.2]},
        
        # GLU variants significantly outperform standard activations in Transformers
        # SwiGLU: Best overall (used in LLaMA, PaLM) - smooth gating
        # GEGLU: Strong alternative - GELU-based gating
        # ReGLU: ReLU-based, faster but slightly worse
        'activation': {'values': ['SwiGLU', 'GEGLU', 'gelu']},
        
        # RMSNorm: Faster than LayerNorm, similar performance (used in LLaMA)
        # LayerNormNoBias: Slightly faster, can improve generalization
        'norm_type': {'values': ['RMSNorm', 'LayerNorm']},
        
        # Critical for non-stationary conflict data
        'use_reversible_instance_norm': {'values': [True]},

        # ============== LOSS FUNCTION ==============
        'loss_function': {'values': ['WeightedPenaltyHuberLoss']},
        
        'zero_threshold': {
            'distribution': 'log_uniform_values',
            'min': 0.01,
            'max': 0.1,
        },
        'delta': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.8,
        },
        # Country-month has more non-zero events - can reduce non_zero_weight
        'non_zero_weight': {
            'distribution': 'uniform',
            'min': 2.0,
            'max': 5.0,  # Lower than priogrid since less sparse
        },
        'false_positive_weight': {
            'distribution': 'uniform',
            'min': 1.0,
            'max': 2.5,
        },
        'false_negative_weight': {
            'distribution': 'uniform',
            'min': 2.0,
            'max': 5.0,
        },
    }

    sweep_config['parameters'] = parameters
    return sweep_config