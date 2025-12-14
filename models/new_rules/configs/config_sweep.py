def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.

    This is a FOCUSED SWEEP around the current best-performing model,
    exploring longer training, longer input, and refined loss penalties
    before pivoting to feature selection.
    """

    sweep_config = {
        'method': 'bayes',
        'name': 'new_rules_nbeats_refinement_sweep',
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 15, # Increased min_iter for longer runs
            'eta': 2
        },
        'metric': {
            'name': 'time_series_wise_msle_mean_sb',
            'goal': 'minimize'
        },
    }

    parameters = {
        # --- Primary Swept Parameters ---
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 4e-4,  # Centered around ~5.8e-4
            'max': 8e-4,
        },
        'early_stopping_patience': {
            'values': [10, 15, 20] # Exploring longer training
        },
        'input_chunk_length': {
            'values': [24, 36, 48] # Exploring longer input range
        },
        'false_negative_weight': {
            'distribution': 'uniform',
            'min': 3.5,   # Exploring more sensitivity (center ~3.88)
            'max': 6.0,
        },
        'false_positive_weight': {
            'distribution': 'uniform',
            'min': 1.0,   # Exploring less sensitivity (center ~1.42)
            'max': 1.8,
        },
        'zero_threshold': {
            'distribution': 'uniform',
            'min': 0.08,  # Exploring lower threshold (center ~0.129)
            'max': 0.15,
        },

        # --- Secondary "look around" parameters ---
        'weight_decay': {
            'distribution': 'uniform',
            'min': 2e-4,  # Centered around ~3.3e-4
            'max': 5e-4,
        },
        'gradient_clip_val': {
            'distribution': 'uniform',
            'min': 0.5,   # Centered around ~0.63
            'max': 0.8,
        },
        'delta': {
            'distribution': 'uniform',
            'min': 0.1,   # Centered around ~0.13
            'max': 0.2,
        },

        # --- Frozen Parameters (based on best run) ---
        'n_epochs': {'values': [300]},
        'batch_size': {'values': [8]},
        'early_stopping_min_delta': {'values': [0.001]},
        'output_chunk_shift': {'values': [0]},
        'dropout': {'values': [0.3]},
        'activation': {'values': ['LeakyReLU']},
        'generic_architecture': {'values': [True]},
        'num_stacks': {'values': [1]},
        'num_blocks': {'values': [1]},
        'num_layers': {'values': [1]},
        'layer_widths': {'values': [64]},
        'feature_scaler': {'values': ['MinMaxScaler']},
        'target_scaler': {'values': ['MinMaxScaler']},
        'log_targets': {'values': [True]},
        'loss_function': {'values': ['WeightedPenaltyHuberLoss']},
        'force_reset': {'values': [True]},
        'lr_scheduler_factor': {'values': [0.463]},
        'lr_scheduler_patience': {'values': [7]},
        'lr_scheduler_min_lr': {'values': [1e-5]},
        'log_features': {'values': [
            [
                "lr_ged_sb", "lr_ged_ns", "lr_ged_os", "lr_acled_sb", "lr_acled_os", 
                "lr_ged_sb_tsum_24", "lr_splag_1_ged_sb", "lr_splag_1_ged_os", "lr_splag_1_ged_ns", 
                "lr_wdi_sm_pop_netm", "lr_wdi_sm_pop_refg_or", "lr_wdi_sp_dyn_imrt_fe_in", "lr_wdi_ny_gdp_mktp_kd",
            ]
        ]},
    }

    sweep_config['parameters'] = parameters
    return sweep_config