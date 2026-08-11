# EXPERIMENT_IN_PROGRESS: violet_visitor is the active HydraNet R&D reference model,
# mid-reconstruction toward the validated v2 gated_NB foundation (Epic #242 S1 #244 /
# S3 #246). Its exact loss_reg / n_posterior_samples are intentionally UNPINNED until
# the roster settles — the datafactory parity tests skip it (C-71/C-87; views-platform/
# views-models#254/#297). Remove this marker + re-pin the parity tests when the roster
# lands.
EXPERIMENT_IN_PROGRESS = True


def get_hp_config():
    """
    violet_visitor — the HydraNet R&D reference model (pgm, africa_me_legacy datafactory).

    STATUS: EXPERIMENT_IN_PROGRESS. This is the legacy committed base — gated forecast
    (gate x body) via a hurdle_shrinkage-composed output, all-cell MSE body, weighted-BCE
    gate. It is mid-reconstruction toward the v2 gated_NB foundation (output_distribution=nb,
    soft_gate, 300 lessons); until that lands (Epic #242 S1 #244 / S3 #246) its exact
    loss_reg / n_posterior_samples stay unpinned. See views-models#254/#297, C-71/C-87.
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
        # gated forecast (gate x body); all-cell body
        'output_distribution': 'hurdle_shrinkage',
        'reg_activation': 'softplus',
        'body_supervision': 'all',  # all-cell body supervision (ADR-065; supersedes the retired point-mask knob)

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
        # gate calibration
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

        # D x K = 4 x 4 = 16 produced draws, matching the other seven roster members
    # (ADR-015 6: the count that matters is the count a model PRODUCES). Was D=8
    # with no head sampler, i.e. 8 produced -- which is why rusty_bucket could not
    # be rewired to the roster: it declares expected_samples_per_model: 16 and the
    # ADR-015 contract refuses a constituent that emits something else.
    #
    # Settled 2026-08-11 by maintainer decision (#146, #372 item 2). This model stays
    # EXPERIMENT_IN_PROGRESS -- its family, composition and seed are still unsettled
    # and its queryset is not yet migrated -- but the SAMPLE COUNT is no longer part
    # of the experiment. It is pinned by the ensemble contract
    # (tests/test_ensemble_configs.py::test_declared_modelset_and_sample_counts_match_reality),
    # not by the roster foundation pins that this model is exempt from.
    'n_posterior_samples': 4,
    'n_head_samples': 4,
        'evaluation_mode': 'stochastic',
        'aggregate_method': 'arithmetic_mean',
        'skip_predictions_delivery': True,
        'min_free_disk_gb': 10.0,
    }
    return hyperparameters
