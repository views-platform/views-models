def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.

    This is a targeted 4-run grid search to test if the best-performing
    1x6 block architecture can be improved by slightly tuning the number
    of blocks and the dropout rate.
    """

    sweep_config = {
        'method': 'grid',
        'name': 'new_rules_grid_test',
        'metric': {
            'name': 'time_series_wise_msle_mean_sb',
            'goal': 'minimize'
        },
    }

    parameters = {
        # --- N-BEATS Architecture ---
        'num_blocks': {'values': [3]},
        'num_stacks': {'values': [2]},
        'dropout': {'values': [0.3]},
        'layer_widths': {'values': [64]},
        'num_layers': {'values': [2]}, # 3 does increase y hat bar
        'activation': {'values': ['LeakyReLU']},
        'generic_architecture': {'values': [True]},
        'batch_norm': {'values': [False]},

        # --- Loss Function ---
        'loss_function': {'values': ['TweedieLoss']},
        'p': {'values': [1.2, 1.5, 1.8]},
        'eps': {'values': [1e-6]},
        'zero_threshold': {'values': [0.05]},
        'non_zero_weight': {'values': [5.0]},
        'false_positive_weight': {'values': [1.0]},
        'false_negative_weight': {'values': [5.0]},

        # --- Trainer & Optimizer ---
        'n_epochs': {'values': [100]},
        'lr': {'values': [0.0003]},
        'optimizer_cls': {'values': ['Adam']},
        'weight_decay': {'values': [0.0003]},
        'gradient_clip_val': {'values': [1]},
        'lr_scheduler_cls': {'values': ['ReduceLROnPlateau']},
        'lr_scheduler_patience': {'values': [7]},
        'lr_scheduler_factor': {'values': [0.46]},
        'lr_scheduler_min_lr': {'values': [0.00001]},
        'early_stopping_patience': {'values': [1]}, # 40 
        'early_stopping_min_delta': {'values': [0.01]},
        
        # --- Data Handling & Input/Output ---
        'input_chunk_length': {'values': [24]},
        'output_chunk_length': {'values': [36]},
        'output_chunk_shift': {'values': [0]},
        'batch_size': {'values': [8]},
        'target_scaler': {'values': [None]},
        'feature_scaler': {'values': [None]},
        'log_targets': {'values': [None]},
        'log_features': {'values': [None]},
        "use_reversible_instance_norm": {'values': [False]}, # darts native
        # 'use_static_covariates': {'values': [True]},

        # --- uncertainty ---
        'mc_dropout': {'values': [False]},
        'num_samples': {'values': [1]},

        # --- Other ---
        'steps': {'values': [[*range(1, 37)]]},
        'force_reset': {'values': [True]},
        'random_state': {'values': [1]},
        'n_jobs': {'values': [-1]},
    }

    sweep_config['parameters'] = parameters
    return sweep_config
