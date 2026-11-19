def get_meta_config():
    """Meta data for tight_canvas — the single-level NEIGHBOURHOOD control (views-baseline#178).

    Documentation except for the parts that are not: `regression_sample_metrics` and
    `evaluation_profile` select the metric set views-evaluation scores with. Every arm of the
    Matryoshka comparison declares the same set, because they are scored against each other and
    a per-arm metric set would score halves of the comparison differently.

    `algorithm` is the one key that says which views-baseline algorithm runs. This one needs
    views-baseline 1.6.0 or later; see requirements.txt.

    ⛔ WHY THIS MODEL EXISTS. The eight nested-doll models draw their levels from
    {cell, near, country, map}, and every level had a simplified single-level control EXCEPT
    `near` — a distance-weighted neighbourhood, not an administrative unit. `PooledConflictology`
    groups cells sharing a code in a column, and an overlapping neighbourhood is not a partition,
    so no column could express it; `dolls: ["near"]` is refused because Matryoshka requires
    "cell". So four of the eight models (nested, local, mapless, stateless) contained a level
    with nothing to compare it against. That gap was found before the 2026-11-17 run, deferred,
    and is closed here.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """

    meta_config = {
        "name": "tight_canvas",
        "algorithm": "NeighbourhoodConflictology",
        "creator": "Simon",
        "level": "pgm",
        "prediction_format": "prediction_frame",
        "regression_targets": ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
        "regression_sample_metrics": ["CRPS", "QS_sample", "MCR_sample", "Brier_rgs_sample"],
        "evaluation_profile": "hydranet_ucdp",
        "rolling_origin_stride": 1,
    }
    return meta_config
