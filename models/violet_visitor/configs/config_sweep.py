def get_sweep_config():

    """
    Contains the configuration for hyperparameter sweeps using WandB.
    This configuration is "operational" so modifying it will change the search strategy, parameter ranges, and other settings for hyperparameter tuning aimed at optimizing model performance.

    Returns:
    - sweep_config (dict): A dictionary containing the configuration for hyperparameter sweeps, defining the methods and parameter ranges used to search for optimal hyperparameters.
    """

    sweep_config = {
    'name': 'violet_visitor_sweep',
    'method': 'grid'
    }

    metric = {
        'name': '36month_mean_squared_error',
        'goal': 'minimize'
        }

    sweep_config['metric'] = metric

    parameters_dict = {

        # ============================================================
        # Ledger / Topology
        # ============================================================
        'time_col': {'value': 'month_id'},
        'id_col': {'value': 'priogrid_gid'},
        'spatial_cols': {'value': ['row', 'col']},
        'identity_cols': {'value': ['month_id', 'priogrid_gid', 'c_id', 'row', 'col']},
        'index_names': {'value': ['month_id', 'priogrid_gid']},
        'features': {'value': ['lr_sb_best', 'lr_ns_best', 'lr_os_best']},
        'input_channels': {'value': 3},
        'row_offset': {'value': 87},
        'col_offset': {'value': 310},
        'height': {'value': 180},
        'width': {'value': 180},

        # ============================================================
        # Model Architecture
        # ============================================================
        'model': {'value': 'HydraBNUNet06_LSTM4'},
        'total_hidden_channels': {'value': 32},
        'dropout_rate': {'values': [0.10, 0.15, 0.20]},        # SWEPT around baseline 0.15
        'window_dim': {'value': 32},
        'output_channels': {'value': 1},
        'weight_init': {'value': 'xavier_norm'},
        'h_init': {'value': 'abs_rand_exp-100'},

        # ============================================================
        # Optimization
        # ============================================================
        'windows_per_lesson': {'value': 3},
        'learning_rate': {'values': [0.0005, 0.001, 0.002]},   # SWEPT around baseline 0.001
        'weight_decay': {'value': 0.1},
        'scheduler': {'value': 'WarmupDecay'},
        'warmup_steps': {'value': 100},
        'clip_grad_norm': {'value': True},
        'torch_seed': {'value': 42},
        'np_seed': {'value': 42},

        # ============================================================
        # Multi-Task Signals
        # ============================================================
        'classification_targets': {'value': ['by_sb_best', 'by_ns_best', 'by_os_best']},
        'regression_targets': {'value': ['lr_sb_best', 'lr_ns_best', 'lr_os_best']},
        'transformations': {'value': {
            'log1p': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
            'asinh': [],
            'identity': [],
        }},
        'derivations': {'value': {
            'binary': [
                {'from': 'lr_sb_best', 'to': 'by_sb_best', 'threshold': 0},
                {'from': 'lr_ns_best', 'to': 'by_ns_best', 'threshold': 0},
                {'from': 'lr_os_best', 'to': 'by_os_best', 'threshold': 0},
            ],
        }},
        'steps': {'value': list(range(1, 37))},
        'time_steps': {'value': 36},

        # ============================================================
        # Loss Functions — HURDLE-NB (#99): truncated-NB body (learnable per-target theta)
        # + class-weighted BCE gate. ALIGNED with config_hyperparameters.py (the canonical run
        # config for the hurdle-NB / coordinate-grounding direction, epic #105); the abandoned
        # tobit/focal stack was removed so a sweep cannot silently benchmark against it (C-155).
        # ============================================================
        'output_distribution': {'value': 'hurdle_nb'},
        'loss_reg': {'value': 'hurdle_nb'},
        'loss_reg_theta_init': {'value': 1.0},
        'learnable_theta': {'value': True},
        'loss_class': {'value': 'weighted_bce'},
        'loss_class_pos_weight': {'value': 10.0},
        'onset_bias_init': {'value': -7.0},
        'freeze_multitask_balancer': {'value': True},

        # ============================================================
        # Scheduled Sampling (ADR-056) — OFF for clean C-113 baseline
        # ============================================================
        'ss_schedule': {'value': 'linear'},
        'ss_warmup_lessons': {'value': 15},
        'ss_epsilon_max': {'value': 0.0},

        # ============================================================
        # Strategy (Curriculum)
        # ============================================================
        'total_lessons': {'value': 40},
        'max_ratio': {'value': 0.95},
        'min_ratio': {'value': 0.05},
        'slope_ratio': {'value': 0.75},
        'roof_ratio': {'value': 0.7},
        'min_events': {'value': 5},
        'sampling_strategy': {'value': 'sigmoid'},
        'sampling_steepness': {'value': 1.0},

        # ============================================================
        # Outbound / Evaluation
        # ============================================================
        'n_posterior_samples': {'value': 16},
        'evaluation_mode': {'value': 'stochastic'},
        'aggregate_method': {'value': 'arithmetic_mean'},
        'skip_predictions_delivery': {'value': True},
        }

    sweep_config['parameters'] = parameters_dict

    return sweep_config
