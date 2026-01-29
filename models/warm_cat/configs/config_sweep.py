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
        'name': 'warm_cat_tide_pgm_balanced_v1',
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 15,
            'eta': 2
        },
        'metric': {
            'name': 'time_series_wise_mse_mean_sb',
            'goal': 'minimize'
        },
    }

    parameters = {
        # ============== TEMPORAL CONFIGURATION ==============
        # input_chunk_length: +0.5 → SHORTER input helps
        'steps': {'values': [[*range(1, 36 + 1)]]},
        'input_chunk_length': {'values': [18, 24, 30]},  # Shorter (was 24-48)
        'output_chunk_shift': {'values': [0]},

        # ============== TRAINING BASICS ==============
        # batch_size: -0.7 → LARGER batches help significantly!
        # early_stopping_patience: -0.5 → HIGHER patience
        # early_stopping_min_delta: +0.2 → smaller threshold
        'batch_size': {'values': [256, 512, 1024]},  # MUCH LARGER (was 32-128)
        'n_epochs': {'values': [250]},
        'early_stopping_patience': {'values': [25, 30, 35]},  # HIGHER (was 15-25)
        'early_stopping_min_delta': {'values': [0.0001, 0.0003]},  # Smaller
        'force_reset': {'values': [True]},

        # ============== OPTIMIZER / SCHEDULER ==============
        # lr: -0.3 → HIGHER LR helps
        # weight_decay: +0.83 → MUCH LOWER weight_decay!
        # lr_scheduler_patience: -0.6 → HIGHER patience
        # gradient_clip_val: -0.63 → HIGHER clipping
        # lr_scheduler_factor: -0.2 → more aggressive decay
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 5e-5,   # Higher (was 1e-5)
            'max': 5e-4,   # Higher (was 2e-4)
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,   # MUCH LOWER (was 1e-4)
            'max': 1e-4,   # MUCH LOWER (was 1e-2)
        },
        'lr_scheduler_factor': {
            'distribution': 'uniform',
            'min': 0.1,    # More aggressive (was 0.3)
            'max': 0.4,    # More aggressive (was 0.6)
        },
        'lr_scheduler_patience': {'values': [8, 10, 12]},  # HIGHER (was 5-10)
        'lr_scheduler_min_lr': {'values': [1e-7]},
        'gradient_clip_val': {
            'distribution': 'uniform',
            'min': 0.8,    # HIGHER (was 0.3)
            'max': 1.5,    # HIGHER (was 0.8)
        },

        # ============== SCALING ==============
        # RobustScaler as default fallback for unmapped features
        # feature_scaler_map assigns optimal scalers based on feature characteristics:
        # - AsinhTransform: Zero-inflated counts (fatalities) and heavily skewed data
        # - MinMaxScaler: Bounded or already-normalized features
        # - SqrtTransform: Mortality rates (positive, moderate skew)
        # - LogTransform: Distance/population features with large positive skew
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
        # num_encoder_layers: -0.5 → MORE encoder layers help
        # num_decoder_layers: +0.26 → fewer decoder layers
        # hidden_size: +0.4 → SMALLER hidden_size
        # decoder_output_dim: -0.2 → larger helps
        'num_encoder_layers': {'values': [2, 3, 4]},  # MORE (was 1-2)
        'num_decoder_layers': {'values': [1, 2]},  # Fewer/same
        'decoder_output_dim': {'values': [16, 24, 32]},  # Slightly larger
        'hidden_size': {'values': [24, 32, 48]},  # SMALLER (was 32-128)
        
        # temporal_hidden_size_past: -0.7 → LARGER helps significantly!
        # temporal_hidden_size_future: -0.6 → LARGER helps
        # temporal_decoder_hidden: -0.14 → larger helps
        # temporal_width_past: +0.2 → smaller
        # temporal_width_future: +0.6 → SMALLER
        'temporal_width_past': {'values': [2, 3, 4]},  # Smaller (was 2-6)
        'temporal_width_future': {'values': [2, 3]},  # SMALLER (was 2-6)
        'temporal_hidden_size_past': {'values': [32, 48, 64]},  # LARGER (was 8-32)
        'temporal_hidden_size_future': {'values': [32, 48, 64]},  # LARGER (was 8-32)
        'temporal_decoder_hidden': {'values': [32, 48, 64]},  # Larger (was 16-64)
        
        # Regularization & normalization
        # use_layer_norm: +0.5 → disabling may help
        # dropout: +0.5 → LOWER dropout
        'use_layer_norm': {'values': [False]},  # Disabled (was hurting)
        'dropout': {'values': [0.1, 0.15, 0.2]},  # LOWER (was 0.3-0.5)
        'use_static_covariates': {'values': [False]},
        'use_reversible_instance_norm': {'values': [False]},

        # ============== LOSS FUNCTION ==============
        # delta: +0.94 → CRITICAL: MUCH LOWER delta needed!
        # false_negative_weight: -0.218 → HIGHER helps
        # false_positive_weight: +0.5 → LOWER
        # non_zero_weight: +0.02 → near zero importance
        'loss_function': {'values': ['WeightedPenaltyHuberLoss']},
        
        'zero_threshold': {'values': [0.05, 0.1]},  # Simplified
        
        # delta: +0.94 → CRITICAL: MUCH LOWER (was 0.05-0.5)
        'delta': {
            'distribution': 'uniform',
            'min': 0.01,
            'max': 0.08,  # MUCH LOWER - near pure L1
        },
        
        # non_zero_weight: +0.02 → near zero importance, keep moderate
        'non_zero_weight': {
            'distribution': 'uniform',
            'min': 4.0,
            'max': 8.0,  # Slightly reduced
        },
        
        # false_positive_weight: +0.5 → LOWER
        'false_positive_weight': {
            'distribution': 'uniform',
            'min': 0.5,   # Lower (was 1.0)
            'max': 1.5,   # Lower (was 2.5)
        },
        
        # false_negative_weight: -0.218 → HIGHER helps
        'false_negative_weight': {
            'distribution': 'uniform',
            'min': 8.0,   # Higher (was 5.0)
            'max': 20.0,  # Higher (was 15.0)
        },
    }

    sweep_config['parameters'] = parameters
    return sweep_config