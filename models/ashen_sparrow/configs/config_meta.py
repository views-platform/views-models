def get_meta_config():
    """Meta data for ashen_sparrow — a distributional baseline member of epic #595.

    Documentation except for the parts that are not: `regression_sample_metrics`
    and `evaluation_profile` select the metric set views-evaluation scores with.
    All sixteen #595 members declare the same set, because they pool into one
    ensemble and a per-member metric set would score halves of it differently.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """

    meta_config = {
        "name": "ashen_sparrow",
        "algorithm": "ConflictologyModel",
        "creator": "Simon",
        "level": "pgm",
        "prediction_format": "prediction_frame",
        "regression_targets": ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
        "regression_sample_metrics": ["CRPS", "QS_sample", "MCR_sample", "Brier_rgs_sample"],
        "evaluation_profile": "hydranet_ucdp",
        "rolling_origin_stride": 1,
    }
    return meta_config
