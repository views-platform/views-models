def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    Optimized for TiDEModel on zero-inflated conflict fatalities data at priogrid-month level.
    
    TiDE (Time-series Dense Encoder) Architecture Notes:
    - Uses MLPs for encoding past and future covariates
    - Temporal projections compress time dimension before dense layers
    - Layer normalization critical for stability with sparse data
    - Reversible instance normalization helps with non-stationary conflict patterns
    
    Priogrid-specific considerations:
    - Heavily zero-inflated (>95% zeros in most cells)
    - Large batch sizes for stable gradient estimates across sparse data
    - Higher dropout for regularization with many similar zero-dominated series
    - Aggressive false negative weighting to catch rare conflict events
    - Smaller model capacity to prevent overfitting on sparse signals
    
    Returns:
    - sweep_config (dict): Configuration for hyperparameter sweeps.
    """

    sweep_config = {
        'method': 'bayes',
        'name': 'warm_cat_tide_pgm',
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 20,  # More iterations needed for large-scale convergence
            'eta': 3  # More aggressive pruning for faster sweeps
        },
        'metric': {
            'name': 'time_series_wise_mse_mean_sb',
            'goal': 'minimize'
        },
    }

    parameters = {
        # ============== TEMPORAL CONFIGURATION ==============
        'steps': {'values': [[*range(1, 36 + 1)]]},
        # Shorter input chunks for memory efficiency with large series
        'input_chunk_length': {'values': [24, 36, 48]},
        'output_chunk_shift': {'values': [0]},

        # ============== TRAINING BASICS ==============
        # Smaller batches for memory efficiency (~80GB RAM)
        'batch_size': {'values': [32, 64, 128]},
        'n_epochs': {'values': [200]},  # Fewer epochs, more data per epoch
        'early_stopping_patience': {'values': [15, 20, 25]},  # More patience for noisy loss
        'early_stopping_min_delta': {'values': [0.0005, 0.001]},  # Finer convergence detection
        'force_reset': {'values': [True]},

        # ============== OPTIMIZER / SCHEDULER ==============
        # Lower learning rates for stability with large batches
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 2e-4,
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-4,
            'max': 1e-2,  # Stronger regularization for sparse data
        },
        'lr_scheduler_factor': {
            'distribution': 'uniform',
            'min': 0.3,
            'max': 0.6,
        },
        'lr_scheduler_patience': {'values': [5, 7, 10]},  # More patience between reductions
        'lr_scheduler_min_lr': {'values': [1e-7]},
        'gradient_clip_val': {
            'distribution': 'uniform',
            'min': 0.3,
            'max': 0.8,  # Tighter clipping for sparse gradient spikes
        },

        # ============== SCALING ==============
        # RobustScaler as default fallback for unmapped features
        # feature_scaler_map assigns optimal scalers based on feature characteristics:
        # - AsinhTransform: Zero-inflated counts (fatalities) and heavily skewed data
        # - MinMaxScaler: Bounded or already-normalized features
        # - SqrtTransform: Mortality rates (positive, moderate skew)
        # - LogTransform: Distance/population features with large positive skew
        'feature_scaler': {'values': [None]},
        'target_scaler': {'values': ['AsinhTransform->MinMaxScaler', 'RobustScaler->MinMaxScaler']},
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
        # Smaller capacity to prevent overfitting on sparse priogrid data
        'num_encoder_layers': {'values': [1, 2]},
        'num_decoder_layers': {'values': [1, 2]},
        'decoder_output_dim': {'values': [8, 16, 32]},
        'hidden_size': {'values': [32, 64, 128]},  # Smaller for 60k sparse series
        
        # Temporal width controls information compression
        'temporal_width_past': {'values': [2, 4, 6]},  # Smaller for memory
        'temporal_width_future': {'values': [2, 4, 6]},
        'temporal_hidden_size_past': {'values': [8, 16, 32]},
        'temporal_hidden_size_future': {'values': [8, 16, 32]},
        'temporal_decoder_hidden': {'values': [16, 32, 64]},
        
        # Regularization & normalization - stronger for sparse data
        'use_layer_norm': {'values': [True, False]},  # Critical for stability
        'dropout': {'values': [0.3, 0.4, 0.5]},  # Higher dropout for sparse data
        'use_static_covariates': {'values': [False]},  # Disable for memory with 60k series
        'use_reversible_instance_norm': {'values': [False]},  # Critical for non-stationary conflict

        # ============== LOSS FUNCTION ==============
        # WeightedPenaltyHuberLoss optimized for heavily zero-inflated priogrid data
        'loss_function': {'values': ['WeightedPenaltyHuberLoss']},
        
        # Zero threshold: higher for priogrid where >95% are zeros
        'zero_threshold': {
            'distribution': 'uniform',
            'min': 0.05,
            'max': 0.2,
        },
        # Smaller delta for less outlier robustness (we want to learn from rare events)
        'delta': {
            'distribution': 'log_uniform_values',
            'min': 0.05,
            'max': 0.5,
        },
        # Higher non-zero weight to focus on the rare conflict events
        'non_zero_weight': {
            'distribution': 'uniform',
            'min': 5.0,
            'max': 15.0,
        },
        # Lower false positive weight (false positives less costly than missing events)
        'false_positive_weight': {
            'distribution': 'uniform',
            'min': 1.0,
            'max': 2.5,
        },
        # Much higher false negative weight - missing a conflict event is critical
        'false_negative_weight': {
            'distribution': 'uniform',
            'min': 5.0,
            'max': 15.0,
        },
    }

    sweep_config['parameters'] = parameters
    return sweep_config