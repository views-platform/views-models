def get_meta_config():
    meta_config = {
        "name": "waking_dream",
        "algorithm": "MixtureBaseline",
        "regression_targets": ["synth_target"],
        "level": "pgm",
        "creator": "synthetic_test",
        "prediction_format": "prediction_frame",
        "rolling_origin_stride": 1,
        "regression_sample_metrics": ["CRPS"],
    }
    return meta_config
