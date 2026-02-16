def get_sweep_config():
    """
    Harmonized sweep configuration for hot_stream (TFT).
    Adapted from novel_heuristics general configuration principles.
    Targeting ~20 trials with Bayesian optimization.
    """

    sweep_config = {
        'method': 'bayes',
        'name': 'hot_stream_harmonized',
        'metric': {
            'name': 'time_series_wise_msle_mean_sb',
            'goal': 'minimize'
        },
    }

    parameters = {
        # --- Harmonized Training Basics (from novel_heuristics) ---
        'batch_size': {'values': [8, 16]},
        'n_epochs': {'values': [150]},
        'early_stopping_patience': {'values': [15]},
        'early_stopping_min_delta': {'values': [0.01]},
        'lr': {
            'distribution': 'uniform',
            'min': 0.0001,
            'max': 0.0005,
        },
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
        'false_negative_weight': {'values': [5.0, 10.0]},
        'non_zero_weight': {'values': [5.0, 10.0]},

        # --- TFT Specific Architecture (Focused Search) ---
        'hidden_size': {'values': [64, 128, 256]},
        'lstm_layers': {'values': [1, 2]},
        'num_attention_heads': {'values': [2, 4]},
        'dropout': {'values': [0.2, 0.3]},
        'full_attention': {'values': [False]},
        'feed_forward': {'values': ['GELU', 'GatedResidualNetwork']},
        'add_relative_index': {'values': [True]},
        'use_static_covariates': {'values': [True]},
        'norm_type': {'values': ['LayerNorm']},
        'skip_interpolation': {'values': [False]},
        'hidden_continuous_size': {'values': [8]},

        # --- Operational Fixed Keys ---
        'steps': {'values': [[*range(1, 37)]]},
        'input_chunk_length': {'values': [24, 36]},
        'output_chunk_length': {'values': [36]},
        'output_chunk_shift': {'values': [0]},
        'num_samples': {'values': [1]},
        'mc_dropout': {'values': [True, False]},
        'random_state': {'values': [1]},
        'force_reset': {'values': [True]},
        'use_reversible_instance_norm': {'values': [False]},
    }

    sweep_config['parameters'] = parameters
    return sweep_config
