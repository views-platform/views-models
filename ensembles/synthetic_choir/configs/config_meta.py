def get_meta_config():
    meta_config = {
        "name": "synthetic_choir",
        "models": ["vertical_dream", "horizontal_dream", "diagonal_dream"],
        "regression_targets": ["synth_target"],
        "level": "pgm",
        "aggregation": "mean",
        "regression_point_metrics": ["MSE"],
        "creator": "synthetic_test",
        "reconciliation": None,
    }
    return meta_config
