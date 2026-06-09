
def get_hp_config():
    """
    C-113 SS-BASELINE + Arm-1 LOSS EXPERIMENT (2026-06-07).

    Two independent things are set here — do NOT read this as an untouched baseline:
      - Scheduled sampling: OFF (ss_epsilon_max=0.0, pure teacher forcing). This is
        the honest zero-point for the autoregressive-runaway investigation — the
        unfixed model, with NO exposure-bias fix mixed in.
      - Regression loss: the Arm-1 HURDLE experiment (lognormal_nll on positive
        cells only), NOT the tobit baseline. See the Loss Functions section below.

    So: SS is the clean baseline, but the loss is an active experiment arm. The
    config_sweep.py "C-113 baseline" sweep is a different setup — it keeps the
    tobit baseline and searches dropout/learning-rate instead.

    Was previously "ensemble member B" (calibration-focused) for the golden_hour
    ensemble (3 HydraNet models × 16 samples = 48), with SS epsilon=0.25 (sweep
    finding: epsilon_max=0.25 → sb MCR=1.01). SS was turned off for the baseline run.

    Settings retained from that config:
    - Seeds: 42/42
    - Sigma: per-target {1.0, 0.75, 0.5}
    - SS warmup: 15 lessons (inert while SS is off)
    - Dropout: 0.15
    - Sampling: sigmoid
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
        # Loss Functions — Arm-1 hurdle: lognormal_nll on positive cells only
        # (the BOUNDED last loss-level experiment — magnitude_calibration dossier
        #  2026-06-08, issue #85). Baseline was Tobit + per-target sigma
        #  (ADR-054/055); Tobit is censored ⇒ mutually exclusive with the hurdle
        #  mask, so the positive-regime loss switches to lognormal_nll with a
        #  SCALAR sigma (a per-target dict is validator-rejected for non-tobit
        #  losses). hurdle_threshold=0 activates the C-45 positive-only mask
        #  (training_engine.py:234-259). ONE mechanism vs the saved baseline.
        # ============================================================
        'loss_reg': 'lognormal_nll',
        'loss_reg_sigma': 0.9,
        'hurdle_threshold': 0,
        'loss_class': 'focal',
        'loss_class_alpha': 0.75,
        'loss_class_gamma': 1.5,
        'onset_bias_init': -7.0,

        # ============================================================
        # Scheduled Sampling (ADR-056) — OFF (clean C-113 baseline, pure teacher forcing)
        # ============================================================
        'ss_schedule': 'linear',
        'ss_warmup_lessons': 15,
        'ss_epsilon_max': 0.0,  # 0 = scheduled sampling OFF — clean baseline (pure teacher forcing)

        # ============================================================
        # Strategy (Curriculum ADR 011/012 Compliance)
        # ============================================================
        'total_lessons': 40,
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
