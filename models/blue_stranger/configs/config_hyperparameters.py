
def get_hp_config():
    """
    Ensemble member C: "CRPS-focused" — uniform sigma + early SS exposure.

    Orthogonal design for golden_hour ensemble (3 HydraNet models × 16 samples = 48).
    Based on sweep finding: uniform sigma=1.0 with SS gives best CRPS (0.140).

    Diversity axes vs other members:
    - Seeds: 99/99 (vs 4/4 and 42/42)
    - Sigma: uniform 1.0 (vs per-target — completely different loss surface)
    - SS warmup: 5 (vs 10 and 15) — early exposure to own predictions
    - Lessons: 250 (vs 200) — deeper convergence
    - Dropout: 0.1 (vs 0.125 and 0.15) — tighter predictions, less uncertainty
    - Sampling: boltzmann (vs threshold and sigmoid) — different spatial focus
    """

    hyperparameters = {

        # ============================================================
        # Ledger / Topology (ADR 007 Compliance)
        # ============================================================
        'time_col': 'month_id',
        'id_col': 'priogrid_gid',
        'spatial_cols': ['row', 'col'],
        'identity_cols': ['month_id', 'priogrid_gid', 'c_id', 'row', 'col'],
        "index_names": ['month_id', 'priogrid_gid'],
        'features': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
        'input_channels': 3,
        'row_offset': 87,
        'col_offset': 310,
        'height': 180,
        'width': 180,

        # ============================================================
        # Model Architecture
        # ============================================================
        'model': 'HydraBNUNet06_LSTM4',
        'total_hidden_channels': 32,
        'dropout_rate': 0.1,
        'window_dim': 32,
        'output_channels': 1,
        'weight_init': 'xavier_norm',
        'h_init': 'abs_rand_exp-100',

        # ============================================================
        # Optimization (ADR 014 Compliance)
        # ============================================================
        'windows_per_lesson': 3,
        'learning_rate': 0.001,
        'weight_decay': 0.1,
        'scheduler': 'WarmupDecay',
        'warmup_steps': 100,
        'clip_grad_norm': True,
        'torch_seed': 99,
        'np_seed': 99,

        # ============================================================
        # Multi-Task Signals (ADR 020 Compliance)
        # ============================================================
        'classification_targets': ['by_sb_best', 'by_ns_best', 'by_os_best'],
        'regression_targets': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],

        'transformations': {
            'log1p': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
            'asinh': [],
            'identity': []
        },

        'derivations': {
            'binary': [
                {'from': 'lr_sb_best', 'to': 'by_sb_best', 'threshold': 0},
                {'from': 'lr_ns_best', 'to': 'by_ns_best', 'threshold': 0},
                {'from': 'lr_os_best', 'to': 'by_os_best', 'threshold': 0},
            ],
        },

        'steps': list(range(1, 37)),
        'time_steps': 36,

        # ============================================================
        # Loss Functions — Tobit + uniform sigma (ADR-054)
        # ============================================================
        'loss_reg': 'tobit',
        'loss_reg_sigma': 1.0,
        'loss_class': 'focal',
        'loss_class_alpha': 0.75,
        'loss_class_gamma': 1.5,
        'onset_bias_init': -7.0,

        # ============================================================
        # Scheduled Sampling (ADR-056) — early exposure
        # ============================================================
        'ss_schedule': 'linear',
        'ss_warmup_lessons': 5,
        'ss_epsilon_max': 0.5,

        # ============================================================
        # Strategy (Curriculum ADR 011/012 Compliance)
        # ============================================================
        'total_lessons': 250,
        'max_ratio': 0.95,
        'min_ratio': 0.05,
        'slope_ratio': 0.75,
        'roof_ratio': 0.7,
        'min_events': 5,
        'sampling_strategy': 'boltzmann',
        'sampling_temperature': 10.0,

        # ============================================================
        # Outbound / Evaluation
        # ============================================================
        'n_posterior_samples': 16,
        'evaluation_mode': 'stochastic',
        'aggregate_method': 'arithmetic_mean',
        'skip_predictions_delivery': True,
    }

    return hyperparameters
