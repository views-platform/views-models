def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.

    Mirrors novel_heuristics sweep config for parity testing.
    """

    sweep_config = {
        'method': 'grid',
        'name': 'shining_codex_01',
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
        'num_layers': {'values': [3]},
        'activation': {'values': ['LeakyReLU']},
        'generic_architecture': {'values': [True]},
        'batch_norm': {'values': [False]},

        # --- Loss Function ---
        'loss_function': {'values': ['WeightedPenaltyHuberLoss']},
        'zero_threshold': {'values': [0.01]},
        'delta': {'values': [0.025]},
        'non_zero_weight': {'values': [7.0]},
        'false_positive_weight': {'values': [1.0]},
        'false_negative_weight': {'values': [10.0]},

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
        'early_stopping_patience': {'values': [1]},
        'early_stopping_min_delta': {'values': [0.01]},

        # --- Data Handling & Input/Output ---
        'input_chunk_length': {'values': [24]},
        'output_chunk_length': {'values': [36]},
        'output_chunk_shift': {'values': [0]},
        'batch_size': {'values': [8]},
        'target_scaler': {'values': ['MinMaxScaler']},
        'feature_scaler': {'values': ['MinMaxScaler']},
        'log_targets': {'values': [True]},
        'log_features': {'values': [None]},
        "use_reversible_instance_norm": {'values': [False]},

        # --- uncertainty ---
        'mc_dropout': {'values': [False]},
        'num_samples': {'values': [1]},

        # --- Other ---
        'steps': {'values': [[*range(1, 37)]]},
        'force_reset': {'values': [True]},
        'random_state': {'values': [1, 2]},
        'n_jobs': {'values': [-1]},
    }

    sweep_config['parameters'] = parameters
    return sweep_config
