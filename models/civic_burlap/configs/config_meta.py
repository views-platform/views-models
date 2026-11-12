def get_meta_config():
    """Meta data for civic_burlap — a smoothed climatology baseline (views-models#695).

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
        "name": "civic_burlap",
        "algorithm": "MatryoshkaConflictology",
        "creator": "Simon",
        "level": "pgm",
        "prediction_format": "prediction_frame",
        "regression_targets": ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
        "regression_sample_metrics": ["CRPS", "QS_sample", "MCR_sample", "Brier_rgs_sample"],
        "evaluation_profile": "hydranet_ucdp",
        "rolling_origin_stride": 1,
    }
    return meta_config
