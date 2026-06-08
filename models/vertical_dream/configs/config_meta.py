def get_meta_config():
    meta_config = {
        "name": "vertical_dream",
        "algorithm": "LocfModel",
        "regression_targets": ["synth_target"],
        "level": "pgm",
        "creator": "synthetic_test",
        "prediction_format": "prediction_frame",
        "rolling_origin_stride": 1,
        "regression_point_metrics": ["MSE"],
    }
    return meta_config
