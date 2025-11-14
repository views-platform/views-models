def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    This configuration is "operational" so modifying it will change the search strategy, parameter ranges, and other settings for hyperparameter tuning aimed at optimizing model performance.

    Returns:
    - sweep_config (dict): A dictionary containing the configuration for hyperparameter sweeps, defining the methods and parameter ranges used to search for optimal hyperparameters.
    """

    sweep_config = {
        'method': 'bayes',
        'name': 'new_rules_nbeats_focus',
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 10,
            'eta': 2
        },
        'metric': {
            'name': 'time_series_wise_msle_mean_sb',
            'goal': 'minimize'
        },
    }

    parameters = {
        # Temporal horizon & context - longer context for rare event patterns
        'steps': {'values': [[*range(1, 36 + 1)]]},
        'input_chunk_length': {'values': [36, 48, 60, 72]},  # Longer to capture rare patterns
        'output_chunk_shift': {'values': [0, 1, 2]},  # Shift prediction horizon

        # Training basics - more epochs to learn rare patterns
        'batch_size': {'values': [32, 64, 96]},  # Smaller batches for better gradient variety
        'n_epochs': {'values': [200, 300, 400]},  # More epochs to learn rare events
        'early_stopping_patience': {'values': [10, 15, 20]},  # More patience for rare events
        'early_stopping_min_delta': {'values': [0.001, 0.005, 0.01]},

        # Optimizer / scheduler - lower learning rates for stability
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 5e-6,  # Lower to prevent missing rare events
            'max': 2e-4,
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-3,
        },
        'lr_scheduler_factor': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.5,
        },
        'lr_scheduler_patience': {'values': [3, 5, 7]},
        'lr_scheduler_min_lr': {'values': [1e-6, 1e-5]},

        # Scaling and transformation - RobustScaler for outlier handling
        'feature_scaler': {'values': ['RobustScaler']},  # Fixed for consistency
        'target_scaler': {'values': ['RobustScaler']},  # Fixed for consistency
        'log_targets': {'values': [True]},  # Fixed to handle zero-inflation
        'log_features': {
            'values': [
                ["lr_ged_sb", "lr_ged_ns", "lr_ged_os", "lr_acled_sb", "lr_acled_os", 
                 "lr_ged_sb_tsum_24", "lr_splag_1_ged_sb", "lr_splag_1_ged_os", "lr_splag_1_ged_ns", 
                 "lr_wdi_sm_pop_netm", "lr_wdi_sm_pop_refg_or", "lr_wdi_sp_dyn_imrt_fe_in", "lr_wdi_ny_gdp_mktp_kd",
                 ]
            ]
        },

        # NBEATS specific architecture - deeper for complex patterns
        'generic_architecture': {'values': [True]},  # Generic for flexibility
        'num_stacks': {'values': [3, 4, 5]},  # More stacks for complexity
        'num_blocks': {'values': [3, 4, 5]},  # More blocks for rare events
        'num_layers': {'values': [3, 4]},  # Deeper networks
        'layer_widths': {'values': [128, 256, 512]},  # Wider for capacity
        'activation': {'values': ['ReLU', 'LeakyReLU', 'GELU']},  # GELU for rare events
        'dropout': {'values': [0.1, 0.2, 0.3]},  # Moderate dropout
        'random_state': {'values': [42, 123, 2023]},
        'force_reset': {'values': [True]},

        # Loss function - WeightedPenaltyHuberLoss for rare events
        'loss_function': {'values': ['WeightedPenaltyHuberLoss']},

        # Loss function parameters - adjusted for log-transformed data and rare events
        'zero_threshold': {
            'distribution': 'uniform',
            'min': 0.1,  # Well above 0 to account for scaling
            'max': 0.5,  # Well below 0.693 to distinguish from smallest non-zero
        },
        'false_positive_weight': {
            'distribution': 'uniform',
            'min': 1.0,  # Lower to avoid over-penalizing rare events
            'max': 3.0,
        },
        'false_negative_weight': {
            'distribution': 'uniform',
            'min': 3.0,  # Higher to penalize missing rare events
            'max': 8.0,
        },
        'non_zero_weight': {
            'distribution': 'uniform',
            'min': 3.0,  # Weight for non-zero events
            'max': 7.0,
        },
        'delta': {
            'distribution': 'log_uniform_values',
            'min': 0.1,  # Higher to focus on absolute errors for rare events
            'max': 1.0,
        },
        
        # Gradient clipping to prevent explosion
        'gradient_clip_val': {
            'distribution': 'uniform',
            'min': 0.5,
            'max': 1.0,
        },
    }

    sweep_config['parameters'] = parameters
    return sweep_config