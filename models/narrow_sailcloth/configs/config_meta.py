def get_meta_config():
    """Meta data for narrow_sailcloth — the moving-window twin of narrow_canvas (views-models#736).

    ⛔ IDENTICAL TO `narrow_canvas` IN EVERY SETTING BUT `algorithm`. That is the entire design.
    `PooledConflictology` pools the last 36 months ending at the last TRAINING month, at every
    rolling origin, so by origin 12 its window is 12 months stale. `RollingPooledConflictology`
    pools the 36 months ending at `train_end + origin` — the information rule the Matryoshka
    models already use. The areas themselves stay fixed at `train_end` in both.

    Why it has to exist: the eight Matryoshka models read history up to each origin and the
    four fixed-window models do not, so a score gap between those two families mixes the
    method with a data advantage and cannot be attributed to either. This model and its twin
    differ ONLY in the information rule, which is what isolates it.

    At origin 0 the two windows are the same window, so this model's scores must come out equal
    to `narrow_canvas`'s. That is the acceptance check before any result here is believed
    (tools/audit/verify_experiment_data.py names the twin; docs/matryoshka_experiment_catalog.md
    section 3).

    Documentation except for the parts that are not: `regression_sample_metrics` and
    `evaluation_profile` select the metric set views-evaluation scores with. All ten arms of
    #695 declare the same set, because they are scored against each other and a per-arm metric
    set would score halves of the comparison differently.

    `algorithm` is the one key that says which of views-baseline's algorithms runs. Both
    members of this pair keep conflictology's chance of zero exactly and pool over the same
    areas; they differ only in where the 36-month window ends.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """

    meta_config = {
        "name": "narrow_sailcloth",
        "algorithm": "RollingPooledConflictology",
        "creator": "Simon",
        "level": "pgm",
        "prediction_format": "prediction_frame",
        "regression_targets": ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
        "regression_sample_metrics": ["CRPS", "QS_sample", "MCR_sample", "Brier_rgs_sample"],
        "evaluation_profile": "hydranet_ucdp",
        "rolling_origin_stride": 1,
    }
    return meta_config
