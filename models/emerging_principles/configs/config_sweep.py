# file: sweep_configs/shrinkage_log1p_test_sweep.py

def get_sweep_config():
    """
    Configuration for testing ShrinkageLoss with log1p transformation
    and a fixed, optimized learning rate on the N-BEATS model.
    """

    sweep_config = {
        "method": "grid",
        "name": "emerging_principles_26",
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # --- N-BEATS Architecture ---
        'num_blocks': {'values': [4]},
        'num_stacks': {'values': [2]},
        'dropout': {'values': [0.3]},
        'layer_widths': {'values': [64]},
        'num_layers': {'values': [3]},
        'activation': {'values': ['LeakyReLU']},
        'generic_architecture': {'values': [True]},



        # --- Loss Function ---
        'loss_function': {'values': ['WeightedPenaltyHuberLoss']},
        'zero_threshold': {'values': [0.01]},
        'delta': {'values': [0.025,]},
        'non_zero_weight': {'values': [7]},
        'false_positive_weight': {'values': [1]},
        'false_negative_weight': {'values': [10.0]},

        # --- Trainer & Optimizer ---
        'n_epochs': {'values': [300]},
        'lr': {'values': [0.0003]},
        'optimizer_cls': {'values': ['Adam']},
        'weight_decay': {'values': [0.0003]},
        'gradient_clip_val': {'values': [1]},
        'lr_scheduler_cls': {'values': ['ReduceLROnPlateau']},
        'lr_scheduler_patience': {'values': [7]},
        'lr_scheduler_factor': {'values': [0.46]},
        'lr_scheduler_min_lr': {'values': [0.00001]},
        'early_stopping_patience': {'values': [40]},
        'early_stopping_min_delta': {'values': [0.01]},
        
        # --- Data Handling & Input/Output ---
        'input_chunk_length': {'values': [36]},
        'output_chunk_length': {'values': [36]},
        'output_chunk_shift': {'values': [0]},
        'batch_size': {'values': [8]},
        'target_scaler': {'values': ['MinMaxScaler']},
        'feature_scaler': {'values': ['MinMaxScaler']},
        'log_targets': {'values': [True]},
        'log_features': {'values': [None]},
        'use_reversible_instance_norm': {'values': [False]}, # darts native
        
        # --- Other ---
        'steps': {'values': [[*range(1, 37)]]},
        'mc_dropout': {'values': [True]},
        'force_reset': {'values': [True]},
        'random_state': {'values': [1, 2]},
        'n_jobs': {'values': [-1]},
    }

    sweep_config['parameters'] = parameters
    return sweep_config
