def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    Optimized for TCNModel (Temporal Convolutional Network) on zero-inflated conflict fatalities data.
    
    TCN Architecture Notes:
    - Causal dilated convolutions capture long-range temporal dependencies
    - Exponentially increasing dilation allows efficient receptive field growth
    - weight_norm stabilizes training, alternative to batch norm
    - Kernel size and dilation_base together determine receptive field
    - Computationally efficient compared to RNN/Transformer
    
    Receptive field = 1 + (kernel_size - 1) * sum(dilation_base^i) for i in 0..num_layers-1
    
    Returns:
    - sweep_config (dict): Configuration for hyperparameter sweeps.
    """

    sweep_config = {
        'method': 'bayes',
        'name': 'teenage_dirtbag_tcn_sweep_week_dylan',
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 12,
            'eta': 2
        },
        'metric': {
            'name': 'time_series_wise_msle_mean_sb',
            'goal': 'minimize'
        },
    }

    parameters = {
        # ============== TEMPORAL CONFIGURATION ==============
        'steps': {'values': [[*range(1, 36 + 1)]]},
        'input_chunk_length': {'values': [36, 48, 60]},
        'output_chunk_length': {'values': [36]},

        # ============== TRAINING BASICS ==============
        'batch_size': {'values': [64, 128, 256]},
        'n_epochs': {'values': [300]},
        'early_stopping_patience': {'values': [10, 12]},
        'early_stopping_min_delta': {'values': [0.001, 0.005]},
        'force_reset': {'values': [True]},
        'save_checkpoints': {'values': [True]},

        # ============== OPTIMIZER / SCHEDULER ==============
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 5e-5,
            'max': 5e-4,
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-3,
        },
        'lr_scheduler_factor': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.4,
        },
        'lr_scheduler_patience': {'values': [3, 5, 7]},
        'lr_scheduler_min_lr': {'values': [1e-6]},
        'gradient_clip_val': {
            'distribution': 'uniform',
            'min': 0.5,
            'max': 1.0,
        },

        # ============== SCALING ==============
        # RobustScaler as default for features not in feature_scaler_map
        # feature_scaler_map applies AsinhTransform to zero-inflated features
        'feature_scaler': {'values': ['RobustScaler']},
        'target_scaler': {'values': ['AsinhTransform', 'RobustScaler']},
        'log_targets': {'values': [False]},
        'feature_scaler_map': {
            'values': [{
                "AsinhTransform": [
                    "ged_sb", "ged_ns", "ged_os", "acled_sb", "acled_os",
                    "ged_sb_tsum_24", "splag_1_ged_sb", "splag_1_ged_os", "splag_1_ged_ns",
                    "wdi_sm_pop_netm", "wdi_sm_pop_refg_or", "wdi_sp_dyn_imrt_fe_in", "wdi_ny_gdp_mktp_kd"
                ]
            }]
        },

        # ============== TCN ARCHITECTURE ==============
        'kernel_size': {'values': [3, 5, 7]},  # Convolution kernel size
        'num_filters': {'values': [64, 128, 256]},  # Number of convolutional filters
        'dilation_base': {'values': [2, 3]},  # Exponential dilation base
        'num_layers': {'values': [None]},  # Auto-compute based on input length
        'dropout': {'values': [0.2, 0.3, 0.4]},
        'weight_norm': {'values': [True]},  # Stabilizes training
        'use_reversible_instance_norm': {'values': [True]},  # Helps with distribution shift

        # ============== LOSS FUNCTION ==============
        'loss_function': {'values': ['WeightedPenaltyHuberLoss']},
        
        'zero_threshold': {
            'distribution': 'uniform',
            'min': 0.01,
            'max': 0.1,
        },
        'delta': {
            'distribution': 'log_uniform_values',
            'min': 0.1,
            'max': 0.8,
        },
        'non_zero_weight': {
            'distribution': 'uniform',
            'min': 4.0,
            'max': 8.0,
        },
        'false_positive_weight': {
            'distribution': 'uniform',
            'min': 1.5,
            'max': 3.0,
        },
        'false_negative_weight': {
            'distribution': 'uniform',
            'min': 2.5,
            'max': 6.0,
        },
    }

    sweep_config['parameters'] = parameters
    return sweep_config