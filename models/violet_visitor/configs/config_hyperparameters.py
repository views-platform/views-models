def get_hp_config():
    """
    FOUNDATION grid (2026-07-17): gated forecast (gate x body), ALL-CELL body, MSE, softplus, BatchNorm fix
    on. Config: output_distribution='hurdle_shrinkage' (gated compose) + NO hurdle_threshold (all-cell body,
    training_engine.py:371-372) + loss_reg='mse'. Swept pos_weight (2.0) x seed (44). 40
    lessons, priogrid_id. body_mask sweep none s44.
    """
    hyperparameters = {
        'time_col': 'month_id',
        'id_col': 'priogrid_id',
        'spatial_cols': ['row', 'col'],
        'identity_cols': ['month_id', 'priogrid_id', 'c_id', 'row', 'col'],
        "index_names": ['month_id', 'priogrid_id'],
        'features': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
        'static_channels': [],
        'input_channels': 3,
        'row_offset': 87,
        'col_offset': 310,
        'height': 180,
        'width': 180,

        'model': 'HydraBNUNet06_LSTM4',
        'total_hidden_channels': 32,
        'dropout_rate': 0.15,
        'window_dim': 32,
        'output_channels': 1,
        'weight_init': 'xavier_norm',
        'h_init': 'abs_rand_exp-100',
        # gated forecast (gate x body); all-cell body via NO hurdle_threshold
        'output_distribution': 'hurdle_shrinkage',
        'reg_activation': 'softplus',
        'body_mask': 'none',
        # (hurdle_threshold intentionally UNSET -> body trains on ALL cells)

        'windows_per_lesson': 3,
        'learning_rate': 0.001,
        'weight_decay': 0.1,
        'scheduler': 'WarmupDecay',
        'warmup_steps': 100,
        'clip_grad_norm': True,
        'torch_seed': 44,
        'np_seed': 44,
        'freeze_multitask_balancer': True,

        'classification_targets': ['by_sb_best', 'by_ns_best', 'by_os_best'],
        'regression_targets': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
        'transformations': {'log1p': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'], 'asinh': [], 'identity': []},
        'derivations': {'binary': [
            {'from': 'lr_sb_best', 'to': 'by_sb_best', 'threshold': 0},
            {'from': 'lr_ns_best', 'to': 'by_ns_best', 'threshold': 0},
            {'from': 'lr_os_best', 'to': 'by_os_best', 'threshold': 0},
        ]},
        'steps': list(range(1, 37)),
        'time_steps': 36,

        # all-cell body, plain MSE
        'loss_reg': 'mse',
        # the swept gate knob
        'loss_class': 'weighted_bce',
        'loss_class_pos_weight': 2.0,
        'onset_bias_init': -7.0,

        'ss_schedule': 'linear',
        'ss_warmup_lessons': 15,
        'ss_epsilon_max': 0.0,

        'total_lessons': 40,
        'max_ratio': 0.95,
        'min_ratio': 0.05,
        'slope_ratio': 0.75,
        'roof_ratio': 0.7,
        'min_events': 5,
        'sampling_strategy': 'sigmoid',
        'sampling_steepness': 1.0,

        'n_posterior_samples': 8,
        'evaluation_mode': 'stochastic',
        'aggregate_method': 'arithmetic_mean',
        'skip_predictions_delivery': True,
        'min_free_disk_gb': 10.0,
    }
    return hyperparameters
