
def get_hp_config():
    """
    Ensemble member B: "Calibration-focused" — conservative SS for MCR near 1.0.

    Orthogonal design for golden_hour ensemble (3 HydraNet models × 16 samples = 48).
    Based on sweep finding: epsilon_max=0.25 delivers sb MCR=1.01 (nearest to ideal).

    Diversity axes vs other members:
    - Seeds: 42/42 (vs 4/4 and 99/99)
    - Sigma: per-target {1.0, 0.75, 0.5} (same as member A, different from C)
    - SS epsilon: 0.25 (vs 0.5 and 0.5) — less self-prediction, better calibration
    - SS warmup: 15 (vs 10 and 5) — longer pure teacher forcing
    - Dropout: 0.15 (vs 0.125 and 0.1) — more regularization, wider posteriors
    - Sampling: sigmoid (vs threshold) — different spatial anchor selection
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
        'dropout_rate': 0.15,
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
        'torch_seed': 42,
        'np_seed': 42,

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
        # Loss Functions — Tobit + per-target sigma (ADR-054/055)
        # ============================================================
        'loss_reg': 'tobit',
        'loss_reg_sigma': {
            'lr_sb_best': 1.0,
            'lr_ns_best': 0.75,
            'lr_os_best': 0.5,
        },
        'loss_class': 'focal',
        'loss_class_alpha': 0.75,
        'loss_class_gamma': 1.5,
        'onset_bias_init': -7.0,

        # ============================================================
        # Scheduled Sampling (ADR-056) — conservative for calibration
        # ============================================================
        'ss_schedule': 'linear',
        'ss_warmup_lessons': 15,
        'ss_epsilon_max': 0.25,

        # ============================================================
        # Strategy (Curriculum ADR 011/012 Compliance)
        # ============================================================
        'total_lessons': 200,
        'max_ratio': 0.95,
        'min_ratio': 0.05,
        'slope_ratio': 0.75,
        'roof_ratio': 0.7,
        'min_events': 5,
        'sampling_strategy': 'sigmoid',
        'sampling_steepness': 1.0,

        # ============================================================
        # Outbound / Evaluation
        # ============================================================
        'n_posterior_samples': 16,
        'evaluation_mode': 'stochastic',
        'aggregate_method': 'arithmetic_mean',
        'skip_predictions_delivery': True,
    }

    return hyperparameters
