def get_meta_config():
    meta_config = {
        "name": "horizontal_dream",
        "algorithm": "LocfModel",
        "regression_targets": ["synth_target"],
        "level": "pgm",
        "creator": "synthetic_test",
        "prediction_format": "prediction_frame",
        "evaluation_mode": "point",
        "rolling_origin_stride": 1,
        "regression_point_metrics": ["MSE"],
    }
    return meta_config
