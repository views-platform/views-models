def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    Optimized for TiDEModel on zero-inflated conflict fatalities data at priogrid-month level.
    
    TiDE (Time-series Dense Encoder) Architecture Notes:
    - Uses MLPs for encoding past and future covariates
    - Temporal projections compress time dimension before dense layers
    - Layer normalization may hurt for sparse priogrid data
    - Reversible instance normalization helps with non-stationary conflict patterns
    
    Parameter Importance Analysis (10 sweeps vs MSE):
    - delta: +0.94 → CRITICAL: MUCH LOWER delta needed!
    - batch_size: -0.7 → LARGER batches help significantly
    - weight_decay: +0.83 → MUCH LOWER weight_decay
    - temporal_hidden_size_past/future: -0.7/-0.6 → LARGER temporal hidden
    - gradient_clip_val: -0.63 → HIGHER clipping
    - lr_scheduler_patience: -0.6 → HIGHER patience
    - num_encoder_layers: -0.5 → MORE encoder layers
    - early_stopping_patience: -0.5 → HIGHER patience
    - target_scaler AsinhTransform: -0.4 → USE AsinhTransform (fixed)
    - use_layer_norm: +0.5 → try disabling
    - dropout: +0.5 → LOWER dropout
    - false_negative_weight: -0.218 → HIGHER helps
    
    Returns:
    - sweep_config (dict): Configuration for hyperparameter sweeps.
    """

    sweep_config = {
        'method': 'bayes',
        'name': 'warm_cat_tide_pgm_balanced_v3_mtd',
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 10,
            'eta': 2
        },
        'metric': {
            'name': 'time_series_wise_mtd_mean_sb',
            'goal': 'minimize'
        },
    }

    parameters = {
            # ============== TEMPORAL CONFIGURATION ==============
            'steps': {'values': [[*range(1, 36 + 1)]]},
            'input_chunk_length': {'values': [36, 48, 72]},  # Slightly shorter
            'output_chunk_shift': {'values': [0]},

            # ============== TRAINING BASICS ==============
            # batch_size: +0.83 importance → CRITICAL: MUCH smaller batches!
            # early_stopping_patience: -0.34 → higher patience helps
            # early_stopping_min_delta: +0.24 → smaller threshold needed
            'batch_size': {'values': [64, 128, 256, 512, 1024]},
            'n_epochs': {'values': [100]},
            'early_stopping_patience': {'values': [6]},  # HIGHER (was 18-25)
            'early_stopping_min_delta': {'values': [0.001]},
            'force_reset': {'values': [True]},

            # ============== OPTIMIZER / SCHEDULER ==============
            # lr: -0.53 importance → HIGHER LR is better for TiDE!
            # weight_decay: -0.4 → HIGHER weight_decay helps
            'lr': {
                'distribution': 'log_uniform_values',
                'min': 5e-5,   # Higher (was 1e-5)
                'max': 2e-3,   # Higher (was 2e-4)
            },
            'weight_decay': {
                'distribution': 'uniform',
                'min': 5e-4,   # MUCH HIGHER (was 1e-5)
                'max': 5e-3,   # MUCH HIGHER (was 5e-4)
            },
            'lr_scheduler_factor': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 0.25,
            },
            'lr_scheduler_patience': {'values': [4]},
            'lr_scheduler_min_lr': {'values': [1e-7]},
            # gradient_clip_val: -0.076 → slightly higher helps
            'gradient_clip_val': {
                'distribution': 'uniform',
                'min': 0.01,
                'max': 1.2,  # Slightly higher range
            },

            # ============== SCALING ==============
            'feature_scaler': {'values': [None]},
            # target_scaler AsinhTransform: -0.4 → USE AsinhTransform (fixed)
            'target_scaler': {'values': ['AsinhTransform->MinMaxScaler']},  # Fixed (RobustScaler was +0.4 worse)
            'feature_scaler_map': {
                'values': [{
                    # Zero-inflated conflict counts - asinh handles zeros and extreme spikes
                    "AsinhTransform->MinMaxScaler": [
                        "lr_ged_sb",
                        "lr_ged_sb_splag_1", "lr_ged_ns_splag_1", "lr_ged_os_splag_1",
                    ],
                    # Spatiotemporal distance features (continuous, positive, moderate skew)
                    "SqrtTransform->MinMaxScaler": [
                        "lr_sptime_dist_k1_ged_sb", "lr_sptime_dist_k1_ged_os", "lr_sptime_dist_k1_ged_ns",
                        "lr_sptime_dist_k10_ged_sb", "lr_sptime_dist_k10_ged_os", "lr_sptime_dist_k10_ged_ns",
                        "lr_sptime_dist_k001_ged_sb", "lr_sptime_dist_k001_ged_os", "lr_sptime_dist_k001_ged_ns",
                        # Mortality rate (positive, moderate skew)
                        "lr_imr_mean",
                    ],
                    # Distance and population features (large positive skew, log-normal-ish)
                    "LogTransform->MinMaxScaler": [
                        "lr_dist_diamsec",
                        "lr_ttime_mean",
                        "lr_bdist3",
                        "lr_capdist",
                        "lr_pop_gpw_sum",
                    ],
                }]
            },

            # ============== TiDE ARCHITECTURE ==============
            # num_encoder_layers: +0.06 → slightly fewer is fine
            'num_encoder_layers': {'values': [1, 2, 4, 6]},  # Simplified
            'num_decoder_layers': {'values': [1, 2, 3, 4]},
            'decoder_output_dim': {'values': [16, 32, 48, 64]},
            'hidden_size': {'values': [8, 16, 32, 64, 128, 192, 256]},
            
            # temporal_width_future: -0.2 → larger values help
            # temporal_hidden_size_past: -0.09 → larger values help
            'temporal_width_past': {'values': [4, 6, 8]},
            'temporal_width_future': {'values': [6, 8, 10]},  # Larger (was 4-8)
            'temporal_hidden_size_past': {'values': [48, 64, 80]},  # Larger (was 32-64)
            'temporal_hidden_size_future': {'values': [32, 48, 64]},
            'temporal_decoder_hidden': {'values': [32, 64, 96, 128, 256]},
            
            # Regularization & normalization
            # dropout: +0.01 → near zero importance, keep moderate
            'use_layer_norm': {'values': [True, False]},
            'dropout': {'values': [0.25, 0.3, 0.35, 0.45]},  # Moderate (was 0.35-0.45)
            'use_static_covariates': {'values': [True, False]},
            'use_reversible_instance_norm': {'values': [False]},

            # ============== LOSS FUNCTION ==============
            # non_zero_weight: +0.4 → LOWER values are better!
            # delta: +0.4 → LOWER delta is better
            # false_negative_weight: +0.124 → slightly lower is better
            # false_positive_weight: +0.03 → near zero importance
            'loss_function': {'values': ['WeightedPenaltyHuberLoss']},
            
            'zero_threshold': {
                'distribution': 'log_uniform_values',
                'min': 0.001,   
                'max': 0.15,   
            },
            
            # delta: +0.4 importance → LOWER is better
            'delta': {
                'distribution': 'uniform',
                'min': 0.01,   # Lower (was 0.02) - more sensitive
                'max': 0.15,   # Lower (was 0.08) - correlation says lower is much better
            },
            
            # non_zero_weight: +0.4 importance → LOWER is better
            'non_zero_weight': {
                'distribution': 'uniform',
                'min': 1.0,    # Slightly higher - non-zero is precious signal
                'max': 20.0,   # Higher max - emphasize rare conflict events
            },
            
            # false_positive_weight: +0.03 → near zero, keep low-moderate
            'false_positive_weight': {
                'distribution': 'uniform',
                'min': 0.5,
                'max': 20.0,
            },
            
            # false_negative_weight: +0.124 → slightly lower is better
            'false_negative_weight': {
                'distribution': 'uniform',
                'min': 0.5,   # Lower (was 5.0)
                'max': 20.0,   # Lower (was 12.0)
            },
        }

    sweep_config['parameters'] = parameters
    return sweep_config