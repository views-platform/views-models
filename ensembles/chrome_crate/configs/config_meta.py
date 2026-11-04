def get_meta_config():
    """Meta for chrome_crate — the cm ensemble of epic #595, chrome_box's country twin.

    This is the side that is reconciled *against*, so it declares `reconciliation: None`
    and **no `reconcile_with`**: the reconciling side names its partner and the partner
    does not name back (`cruel_summer`'s shape, and what `tests/test_ensemble_configs.py`
    expects — a `reconcile_with` here would have to name an ensemble too).

    Same eight settings as chrome_box in the same order, at `cm` instead of `pgm`. Equal
    draw counts on both sides are what puts the reconciliation in ALIGNED_DRAWS rather
    than POINT_BROADCAST: the mode is chosen from the cm frame's `sample_count`, not
    configured, and an incompatible count fails loud inside the reconciler's validation.

    ⚠️ No `evaluation_profile`, for the reason given in chrome_box's config_meta: the
    `hydranet_ucdp` profile is byte-identical to the `base` default, and declaring it
    would incur a C-132 occurrence-gate obligation these regression-only members cannot
    honestly meet.

    Returns:
    - meta_config (dict): A dictionary containing ensemble meta configuration.
    """
    meta_config = {
        "name": "chrome_crate",
        "regression_targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
        "level": "cm",
        "aggregation": "concat",
        "regression_sample_metrics": ["CRPS", "QS_sample", "MCR_sample", "Brier_rgs_sample"],
        "creator": "Simon",
        "reconciliation": None,
    }
    return meta_config
