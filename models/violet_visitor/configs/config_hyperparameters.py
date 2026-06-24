def get_hp_config():
    """
    HURDLE-NB overnight runs (2026-06-11, ZINB epic #102, decision A).

    The first config that *turns on* the hurdle-NB stack (#98-#101):
      - output_distribution='hurdle_nb' -> softplus mu count-space head (#100)
      - loss_reg='hurdle_nb'            -> TruncatedNBLoss body, learnable per-target theta (#99)
      - loss_class='weighted_bce'       -> proper class-weighted gate (replaces focal; #99)
      - freeze_multitask_balancer=True  -> equal-weight sum of the per-target NLLs (D2/C-141)
      - scheduled sampling OFF          -> we are testing the head, not rollout training
    Inference emits the EXACT hurdle mean E[y]=P(y>0)*mu/(1-NB0(mu,theta)) (#101).

    Run parameters are fixed literals below (standard config style, no env vars):
    torch_seed/np_seed=42, loss_reg_theta_init=1.0, loss_class_pos_weight=10.0,
    total_lessons=40.
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
        # No-coords hurdle floor (R4 baseline) — coords parked (R5/C-167..169; side-probe net-negative).
        'static_channels': [],
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
        # Hurdle-NB head (#100): softplus mu count-space head. (reg_activation flag omitted ⇒
        # default softplus under hurdle_nb — Exp-B-only override removed.)
        'output_distribution': 'hurdle_nb',

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
        # D2/C-141: the per-target hurdle-NB NLLs are summed equal-weight (balancer frozen).
        'freeze_multitask_balancer': True,

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
        # Loss Functions — HURDLE-NB (#99): truncated-NB body (learnable theta) +
        # class-weighted BCE gate.
        # ============================================================
        # Restored no-coords hurdle floor (R4 baseline). Unused shrinkage/focal params (loss_reg_a/c,
        # loss_class_alpha/gamma) left from overnight runs — harmless (registry reads only selected loss).
        'loss_reg': 'hurdle_nb',
        'loss_reg_a': 258,
        'loss_reg_c': 0.001,
        'loss_reg_theta_init': 1.0,
        'learnable_theta': True,
        'loss_class': 'weighted_bce',
        'loss_class_alpha': 0.75,
        'loss_class_gamma': 1.5,
        'loss_class_pos_weight': 10.0,
        'onset_bias_init': -7.0,

        # ============================================================
        # Scheduled Sampling — OFF (testing the head, not rollout training).
        # ============================================================
        'ss_schedule': 'linear',
        'ss_warmup_lessons': 15,
        'ss_epsilon_max': 0.0,

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
        # Experiment operating point (2026-06-16, #128): 8 samples — gated by the 8-sample OOM
        # check (one run must complete without the C-116 eval-stage OOM). Reduced from 16 as the
        # interim memory workaround; restore to 16 once C-116/#124 is fixed.
        'n_posterior_samples': 8,
        'evaluation_mode': 'stochastic',
        'aggregate_method': 'arithmetic_mean',
        'skip_predictions_delivery': True,
        # #110/C-154: abort before the ~2.5 GB prediction writes if the volume is short
        # (the S3_seed4 baseline run was truncated by disk-full). Guard added in #107.
        'min_free_disk_gb': 10.0,
    }

    return hyperparameters
