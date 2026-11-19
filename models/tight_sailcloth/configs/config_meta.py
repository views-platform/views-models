def get_meta_config():
    """Meta data for tight_sailcloth — the MOVING-WINDOW neighbourhood control
    (views-baseline#178).

    Documentation except for the parts that are not: `regression_sample_metrics` and
    `evaluation_profile` select the metric set views-evaluation scores with. Every arm of the
    Matryoshka comparison declares the same set, because they are scored against each other and
    a per-arm metric set would score halves of the comparison differently.

    `algorithm` is the one key that differs from `tight_canvas`. Everything else in this model,
    including every hyperparameter, is identical to it — that is the whole design. If a single
    other setting differed, a score gap between the two would no longer measure the information
    rule alone.

    ⛔ THE ACCEPTANCE CHECK THIS MODEL OWES. At the first origin a moving-window model has seen
    exactly what its fixed-window twin has seen, so its predictions must be EQUAL to
    `tight_canvas`'s — byte for byte, not close. `tools/audit/verify_experiment_data.py` declares
    that pair and reports it on every run. Do not believe this model's results before that check
    passes. views-baseline asserts the same identity upstream in
    `tests/test_rolling_pooled_conflictology.py`.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """

    meta_config = {
        "name": "tight_sailcloth",
        "algorithm": "RollingNeighbourhoodConflictology",
        "creator": "Simon",
        "level": "pgm",
        "prediction_format": "prediction_frame",
        "regression_targets": ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
        "regression_sample_metrics": ["CRPS", "QS_sample", "MCR_sample", "Brier_rgs_sample"],
        "evaluation_profile": "hydranet_ucdp",
        "rolling_origin_stride": 1,
    }
    return meta_config
