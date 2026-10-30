def get_meta_config():
    meta_config = {
        "name": "horizontal_dream",
        "algorithm": "LocfModel",
        "regression_targets": ["synth_target"],
        "level": "pgm",
        "creator": "synthetic_test",
        "prediction_format": "prediction_frame",
        "evaluation_mode": "point",
        "aggregate_method": "arithmetic_mean",  # required by the sniffer for point mode (#477)
        "rolling_origin_stride": 1,
        "regression_point_metrics": ["MSE"],
    }
    return meta_config
