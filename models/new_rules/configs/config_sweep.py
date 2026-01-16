def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.

    This is a targeted 4-run grid search to test if the best-performing
    1x6 block architecture can be improved by slightly tuning the number
    of blocks and the dropout rate.
    """

    sweep_config = {
        'method': 'grid',
        'name': 'new_rules_03',
        'metric': {
            'name': 'time_series_wise_msle_mean_sb',
            'goal': 'minimize'
        },
    }

    parameters = {
        # --- Swept Parameters (2x2 grid) ---
        'num_blocks': {'values': [6]},
        'num_stacks': {'values': [1]},
        'dropout': {'values': [0.3]},
        'layer_widths': {'values': [8, 16]},
        'num_layers': {'values': [1]},
        'input_chunk_length': {'values': [24]},
        'lr': {'values': [0.0006]},
        'random_state': {'values': [1]},

        # --- Frozen Parameters (from Best Run #1) ---
        'delta': {'values': [0.13]},
        'steps': {'values': [[*range(1, 37)]]},
        'n_epochs': {'values': [300]},
        'activation': {'values': ['LeakyReLU']},
        'batch_norm': {'values': [False]},
        'batch_size': {'values': [8]},
        'mc_dropout': {'values': [True]},
        'force_reset': {'values': [True]},
        'log_targets': {'values': [True]},
        'log_features': {'values': [None]}, #[[
        #"lr_ged_sb", "lr_ged_ns", "lr_ged_os", 
            #"lr_acled_sb", "lr_acled_os", "lr_acled_sb_count",
            #"lr_ged_sb_tsum_24", "lr_splag_1_ged_sb", "lr_splag_1_ged_os", "lr_splag_1_ged_ns",
            #"lr_wdi_sm_pop_netm", "lr_wdi_sm_pop_refg_or", "lr_wdi_sp_dyn_imrt_fe_in", "lr_wdi_ny_gdp_mktp_kd"
        #]]},
        'weight_decay': {'values': [0.0003]},
        'loss_function': {'values': ['WeightedPenaltyHuberLoss']},
        'optimizer_cls': {'values': ['Adam']},
        'target_scaler': {'values': ['MinMaxScaler']},
        'feature_scaler': {'values': ['MinMaxScaler']},
        'zero_threshold': {'values': [0.13]},
        'non_zero_weight': {'values': [2.5]},
        'lr_scheduler_cls': {'values': ['ReduceLROnPlateau']},
        'gradient_clip_val': {'values': [0.64]},
        'output_chunk_shift': {'values': [0]},
        'lr_scheduler_factor': {'values': [0.46]},
        'lr_scheduler_min_lr': {'values': [0.00001]},
        'output_chunk_length': {'values': [36]},
        'generic_architecture': {'values': [True]},
        'false_negative_weight': {'values': [4]},
        'false_positive_weight': {'values': [1.5]},
        'lr_scheduler_patience': {'values': [7]},
        'early_stopping_patience': {'values': [20]},
        'early_stopping_min_delta': {'values': [0.001]},
    }

    sweep_config['parameters'] = parameters
    return sweep_config
