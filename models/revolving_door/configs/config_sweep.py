def get_sweep_config():

    """
    N-HiTS (Neural Hierarchical Interpolation for Time Series) Sweep Configuration
    
    N-HiTS Architecture Overview:
    - Multi-rate input sampling via pooling (reduces computational cost)
    - Hierarchical interpolation for multi-scale temporal patterns
    - Stack-based architecture (like NBEATS) with blocks per stack
    - Each stack captures different frequency components
    """
    sweep_config = {
        'method': 'bayes',
        'name': 'revolving_door_nhits_balanced_v3_mtd',
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 20,
            'eta': 2
        },
        'metric': {
            'name': 'time_series_wise_mtd_mean_sb',
            'goal': 'minimize'
        },
    }

    parameters = {
        # ============== TEMPORAL CONFIGURATION ==============
        # input_chunk_length: +0.4 importance → shorter is better for MSLE
        'steps': {'values': [[*range(1, 36 + 1)]]},
        'input_chunk_length': {'values': [36, 48]},  # Shorter context (was 48-72)
        'output_chunk_shift': {'values': [0]},
        'random_state': {'values': [67]},
        'mc_dropout': {'values': [True]},

        # ============== N-HiTS ARCHITECTURE ==============
        # num_stacks: -0.6 importance → FEWER stacks = better MSLE
        # 2 stacks (trend + seasonality) is optimal - simpler model generalizes better
        'num_stacks': {'values': [1, 2, 3]},  # FIXED to 2 (was 2-3)
        
        # num_blocks: Blocks per stack (depth within each scale)
        'num_blocks': {'values': [1, 2, 3]},
        
        # num_layers: FC layers per block
        'num_layers': {'values': [2, 3, 4]},
        
        # layer_width: Slightly smaller to avoid overprediction
        'layer_width': {'values': [64, 128, 256]},
        
        # pooling_kernel_sizes: Controls multi-rate input sampling
        # - None = auto-configured based on input_chunk_length (recommended)
        # - Custom: tuple of tuples (num_stacks x num_blocks)
        'pooling_kernel_sizes': {'values': [None]},  # Let Darts auto-configure
        
        # n_freq_downsample: Controls multi-scale interpolation
        # - None = auto-configured based on output_chunk_length (recommended)
        'n_freq_downsample': {'values': [None]},  # Let Darts auto-configure
        
        # max_pool_1d: MaxPool (True) vs AvgPool (False)
        # - MaxPool: captures peak values, good for sparse event detection
        # - AvgPool: smoother, may miss sparse spikes
        'max_pool_1d': {'values': [True, False]},  # MaxPool better for conflict spikes
        
        # activation: Non-linearity between layers
        # - ReLU: fast, good default
        # - GELU: smoother gradients, often better for time series
        'activation': {'values': ['ReLU', 'ELU', 'GELU']},
        
        # dropout: Regularization to prevent overprediction
        'dropout': {'values': [0.05, 0.1, 0.15]},

        # ============== TRAINING BASICS ==============
        "batch_size": {"values": [256, 512, 1024, 2048]},
        'n_epochs': {'values': [100]},  # More epochs since we reduced regularization
        'early_stopping_patience': {'values': [15, 20, 25]},  # More patience for scarce signal
        "early_stopping_min_delta": {"values": [0.00005, 0.0001]},  # Smaller for [0,1] loss scale
        'force_reset': {'values': [True]},

        # ============== OPTIMIZER / SCHEDULER ==============
        # lr: +0.4 importance → keep LR LOW
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 5e-5,
            'max': 1e-3,
        },
        'weight_decay': {'values': [0]},
        'lr_scheduler_factor': {'values': [0.5]},  # Fixed for stability
        'lr_scheduler_patience': {'values': [8]},  # Fixed - consistent plateau detection
        'lr_scheduler_min_lr': {'values': [1e-6]},  # Higher floor
        "gradient_clip_val": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1.5,
        },
        
        # ============== INSTANCE NORMALIZATION ==============
        # use_reversible_instance_norm: Handles distribution shift
        # - False: Recommended unless you specifically trained with it
        # - True: Can help with non-stationary data, but must match train/predict
        'use_reversible_instance_norm': {'values': [False, True]},

        # ============== LOSS FUNCTION ==============
        # BALANCED: Symmetric-ish weights to avoid over/under prediction
        'loss_function': {'values': ['WeightedPenaltyHuberLoss']},
        
        'zero_threshold': {
            'distribution': 'uniform',  # uniform is fine for this small range
            'min': 0.05,   # Safely above 0 noise
            'max': 0.18,   # Just above where 1 fatality lands (~0.11)
        },
        
        # Moderate delta for balanced L1/L2 behavior
        'delta': {
            'distribution': 'uniform',
            'min': 0.8,
            'max': 1.0,  # Full L2
        },
        
        # non_zero_weight: -0.3 importance → moderate values
        'non_zero_weight': {
            'distribution': 'uniform',
            'min': 4.0,
            'max': 7.0,  # Narrower range prevents conflicting gradients
        },
        
        # BALANCED: Similar FN/FP weights to avoid bias
        "false_positive_weight": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1.0,
        },
        
        # Increase FP penalty to curb overprediction
        'false_negative_weight': {
            'distribution': 'uniform',
            'min': 2.0,
            'max': 5.0,  # Narrower - still emphasizes missing conflicts
        },

        # ============== SCALING ==============
        # AsinhTransform->MinMaxScaler for target: handles zeros + bounds output
        # feature_scaler_map: feature-specific scaling based on data characteristics
        'feature_scaler': {'values': [None]},
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

    }

    sweep_config['parameters'] = parameters
    return sweep_config