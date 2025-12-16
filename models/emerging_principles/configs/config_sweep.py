def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.

    This is a DECISIVE EXPERIMENT GRID SWEEP to test if a larger, regularized
    architecture can outperform the simple baseline now that dropout is working.
    """

    sweep_config = {
        'method': 'grid',
        'name': 'emerging_principles_nbeats_blocks_test',
        'metric': {
            'name': 'time_series_wise_msle_mean_sb',
            'goal': 'minimize'
        },
    }

    parameters = {
        # --- Swept Parameters ---
        'dropout': {'values': [0.3]},
        'input_chunk_length': {'values': [24]},
        'layer_widths': {'values': [16]},

        # --- Frozen Parameters (based on best run) ---
        'early_stopping_patience': {'values': [20]}, # Loosened for this experiment

        'lr': {'values': [0.0005873328851386325]},
        'steps': {'values': [[*range(1, 36 + 1, 1)]]},
        'activation': {'values': ['LeakyReLU']},
        'generic_architecture': {'values': [True]},
        'num_stacks': {'values': [2,3,4]},
        'num_blocks': {'values': [3,4,5,6,7,8]},
        'num_layers': {'values': [1]},
        'output_chunk_shift': {'values': [0]},
        'batch_size': {'values': [8]},
        'n_epochs': {'values': [300]},
        'early_stopping_min_delta': {'values': [0.001]},
        'weight_decay': {'values': [0.0003292268280079564]},
        'lr_scheduler_factor': {'values': [0.46300979785707297]},
        'lr_scheduler_min_lr': {'values': [0.00001]},
        'lr_scheduler_patience': {'values': [7]},
        'gradient_clip_val': {'values': [0.6336557913524701]},
        'feature_scaler': {'values': ['MinMaxScaler']},
        'target_scaler': {'values': ['MinMaxScaler']},
        'log_targets': {'values': [True]},
        'log_features': {'values': [
            [
                "lr_ged_sb", "lr_ged_ns", "lr_ged_os",
                "lr_acled_sb", "lr_acled_os",
                "lr_ged_sb_tsum_24",
                "lr_splag_1_ged_sb", "lr_splag_1_ged_os", "lr_splag_1_ged_ns",
                "lr_wdi_sm_pop_netm", "lr_wdi_sm_pop_refg_or",
                "lr_wdi_sp_dyn_imrt_fe_in", "lr_wdi_ny_gdp_mktp_kd",
            ]
        ]},
        'loss_function': {'values': ['WeightedPenaltyHuberLoss']},
        'delta': {'values': [0.129050050430042]},
        'zero_threshold': {'values': [0.12953171739852642]},
        'false_positive_weight': {'values': [1.4269851202559674]},
        'false_negative_weight': {'values': [3.8819100926929138]},
        'non_zero_weight': {'values': [2.504275866632825]},
        'force_reset': {'values': [True]},
        'num_samples': {'values': [1]},
        'mc_dropout': {'values': [True]},
    }

    sweep_config['parameters'] = parameters
    return sweep_config
