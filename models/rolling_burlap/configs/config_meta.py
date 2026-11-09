def get_meta_config():
    """Meta data for rolling_burlap — the moving-window control (views-models#716).

    Identical to `plain_burlap` in every setting; the only difference is `algorithm`.
    `ConflictologyModel` forecasts every rolling origin from one window ending at the last
    TRAINING month, so by origin 12 its information is 12 months stale. `RollingConflictology`
    forecasts origin k from the window ending at train_end + k — the same information rule
    views-stepshifter uses, so no forecast month is ever inside its own window.

    Comparing the two isolates one quantity: how much of conflictology's staleness is just an
    out-of-date window at the origin. Origin 0 is byte-identical to plain_burlap, so only
    evaluation differs. It is also the like-for-like control for MatryoshkaConflictology
    (views-baseline ADR-024, views-models#717).

    Was: a smoothed climatology baseline template (views-models#695).

    Documentation except for the parts that are not: `regression_sample_metrics` and
    `evaluation_profile` select the metric set views-evaluation scores with. All ten arms of
    #695 declare the same set, because they are scored against each other and a per-arm metric
    set would score halves of the comparison differently.

    `algorithm` is the one key that says which of views-baseline 1.1.0's algorithms runs.
    All of them keep conflictology's chance of zero exactly; they differ only in how the
    NON-ZERO values are drawn.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """

    meta_config = {
        "name": "rolling_burlap",
        "algorithm": "RollingConflictology",
        "creator": "Simon",
        "level": "pgm",
        "prediction_format": "prediction_frame",
        "regression_targets": ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
        "regression_sample_metrics": ["CRPS", "QS_sample", "MCR_sample", "Brier_rgs_sample"],
        "evaluation_profile": "hydranet_ucdp",
        "rolling_origin_stride": 1,
    }
    return meta_config
