def get_sweep_config():

    """
    N-HiTS (Neural Hierarchical Interpolation for Time Series) Sweep Configuration
    
    N-HiTS Architecture Overview:
    - Multi-rate input sampling via pooling (reduces computational cost)
    - Hierarchical interpolation for multi-scale temporal patterns
    - Stack-based architecture (like NBEATS) with blocks per stack
    - Each stack captures different frequency components
    
    Key differences from NBEATS:
    - Uses pooling_kernel_sizes for input downsampling per block
    - Uses n_freq_downsample for output interpolation scales
    - Generally faster training with comparable accuracy
    
    For conflict forecasting (zero-inflated, sparse events):
    - Moderate num_stacks (2-3) to capture trend/seasonality/residual
    - Low num_blocks (1-2) per stack to avoid overfitting sparse data
    - Moderate layer_width (256-512) - too wide overfits sparse targets
    - Higher dropout (0.2-0.4) for regularization on imbalanced data
    """
    sweep_config = {
        'method': 'bayes',
        'name': 'revolving_door_nhits_cm_rinF',
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 15,
            'eta': 2
        },
        'metric': {
            'name': 'time_series_wise_msle_mean_sb',
            'goal': 'minimize'
        },
    }

    parameters = {
        # ============== TEMPORAL CONFIGURATION ==============
        # N-HiTS auto-configures pooling/downsampling based on input/output lengths
        # input_chunk_length should be >= 2x output for good multi-scale decomposition
        'steps': {'values': [[*range(1, 36 + 1)]]},
        'input_chunk_length': {'values': [48, 60, 72]},  # 4-6 years of monthly data
        'output_chunk_shift': {'values': [0]},

        # ============== N-HiTS ARCHITECTURE ==============
        # num_stacks: Each stack captures different temporal scales
        # - 2 stacks: trend + seasonality (simpler, less overfitting risk)
        # - 3 stacks: trend + seasonality + residual (default, more expressive)
        'num_stacks': {'values': [2, 3]},
        
        # num_blocks: Blocks per stack (depth within each scale)
        # - Keep low (1-2) for sparse conflict data to avoid overfitting
        'num_blocks': {'values': [1, 2]},
        
        # num_layers: FC layers per block (before forking layers)
        # - 2-3 is typical; more layers = more capacity but slower
        'num_layers': {'values': [2, 3]},
        
        # layer_width: Neurons per FC layer
        # - 256-512 is sweet spot for conflict data
        # - Larger (1024) may overfit sparse events
        'layer_width': {'values': [256, 384, 512]},
        
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
        'max_pool_1d': {'values': [True]},  # MaxPool better for conflict spikes
        
        # activation: Non-linearity between layers
        # - ReLU: fast, good default
        # - GELU: smoother gradients, often better for time series
        'activation': {'values': ['ReLU', 'GELU']},
        
        # dropout: Regularization (critical for sparse conflict data)
        # - Higher values (0.2-0.4) help prevent overfitting to rare events
        'dropout': {
            'distribution': 'uniform',
            'min': 0.15,
            'max': 0.35,
        },

        # ============== TRAINING BASICS ==============
        'batch_size': {'values': [64, 128]},  # N-HiTS handles larger batches well
        'n_epochs': {'values': [300]},
        'early_stopping_patience': {'values': [10, 15]},
        'early_stopping_min_delta': {'values': [0.001]},
        'force_reset': {'values': [True]},

        # ============== OPTIMIZER / SCHEDULER ==============
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 1e-4,
            'max': 1e-3,  # N-HiTS typically uses slightly higher LR than TiDE
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-3,
        },
        'lr_scheduler_factor': {
            'distribution': 'uniform',
            'min': 0.2,
            'max': 0.5,
        },
        'lr_scheduler_patience': {'values': [4, 5, 6]},
        'lr_scheduler_min_lr': {'values': [1e-6]},
        'gradient_clip_val': {
            'distribution': 'uniform',
            'min': 0.5,
            'max': 1.0,
        },
        
        # ============== INSTANCE NORMALIZATION ==============
        # use_reversible_instance_norm: Handles distribution shift
        # - False: Recommended unless you specifically trained with it
        # - True: Can help with non-stationary data, but must match train/predict
        'use_reversible_instance_norm': {'values': [False]},

        # ============== LOSS FUNCTION ==============
        # WeightedPenaltyHuberLoss optimized for zero-inflated conflict data
        'loss_function': {'values': ['WeightedPenaltyHuberLoss']},
        'zero_threshold': {'values': [0.01]},  # Below this = "zero" for loss weighting
        'delta': {
            'distribution': 'uniform',
            'min': 0.3,
            'max': 0.7,
        },
        'non_zero_weight': {
            'distribution': 'uniform',
            'min': 3.0,
            'max': 8.0,
        },
        'false_negative_weight': {
            'distribution': 'uniform',
            'min': 10.0,
            'max': 20.0,  # Penalize missing actual conflict heavily
        },
        'false_positive_weight': {
            'distribution': 'uniform',
            'min': 5.0,
            'max': 12.0,
        },

        # ============== SCALING ==============
        # AsinhTransform->MinMaxScaler for target: handles zeros + bounds output
        # feature_scaler_map: feature-specific scaling based on data characteristics
        'feature_scaler': {'values': [None]},
        'target_scaler': {'values': ['AsinhTransform->MinMaxScaler']},
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

    }

    sweep_config['parameters'] = parameters
    return sweep_config