def get_sweep_config():
    """
    Harmonized sweep configuration for adolecent_slob (TCN).
    Adapted from novel_heuristics general configuration principles.
    Targeting ~20 trials with Bayesian optimization.
    """

    sweep_config = {
        'method': 'grid',
        'name': 'adolecent_slob_grid_test',
        'metric': {
            'name': 'time_series_wise_msle_mean_sb',
            'goal': 'minimize'
        },
    }

    parameters = {
        # --- Harmonized Training Basics (from novel_heuristics) ---
        'batch_size': {'values': [16]},
        'n_epochs': {'values': [150]},
        'early_stopping_patience': {'values': [15]},
        'early_stopping_min_delta': {'values': [0.01]},
        'lr': {'values': [0.0003952169009532478]},
        'weight_decay': {'values': [0.0001]},
        'optimizer_cls': {'values': ['Adam']},
        'gradient_clip_val': {'values': [1.0]},
        'lr_scheduler_cls': {'values': ['ReduceLROnPlateau']},
        'lr_scheduler_factor': {'values': [0.46]},
        'lr_scheduler_patience': {'values': [7]},
        'lr_scheduler_min_lr': {'values': [0.00001]},
        
        # --- Harmonized Scaling & Data ---
        'feature_scaler': {'values': ['MinMaxScaler']},
        'target_scaler': {'values': ['MinMaxScaler']},
        'log_targets': {'values': [True]},
        'log_features': {
            'values': [
                ["lr_ged_sb", "lr_ged_ns", "lr_ged_os", "lr_acled_sb", "lr_acled_os", 
                 "lr_ged_sb_tsum_24", "lr_splag_1_ged_sb", "lr_splag_1_ged_os", "lr_splag_1_ged_ns", 
                 "lr_wdi_sm_pop_netm", "lr_wdi_sm_pop_refg_or", "lr_wdi_sp_dyn_imrt_fe_in", "lr_wdi_ny_gdp_mktp_kd",
                 ]
            ]
        },

        # --- Harmonized Loss Function ---
        'loss_function': {'values': ['WeightedPenaltyHuberLoss']},
        'delta': {'values': [0.025]},
        'zero_threshold': {'values': [0.01]},
        'false_positive_weight': {'values': [1.0]},
        'false_negative_weight': {'values': [10.0]},
        'non_zero_weight': {'values': [5.0]},

        # --- TCN Specific Architecture (Focused Search) ---
        'kernel_size': {'values': [5]},
        'num_filters': {'values': [64]},
        'dilation_base': {'values': [3]},
        'dropout': {'values': [0.3]},
        'weight_norm': {'values': [False]},

        # --- Operational Fixed Keys ---
        'steps': {'values': [[*range(1, 37)]]},
        'input_chunk_length': {'values': [48]},
        'output_chunk_length': {'values': [36]},
        'output_chunk_shift': {'values': [0]},
        'num_samples': {'values': [1]},
        'mc_dropout': {'values': [False]},
        'random_state': {'values': [1, 2]},
        'force_reset': {'values': [True]},
        'use_reversible_instance_norm': {'values': [True]},
    }

    sweep_config['parameters'] = parameters
    return sweep_config
