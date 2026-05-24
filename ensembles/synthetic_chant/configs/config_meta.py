def get_meta_config():
    meta_config = {
        "name": "synthetic_chant",
        "models": ["lucid_dream", "vivid_dream", "waking_dream"],
        "regression_targets": ["synth_target"],
        "level": "pgm",
        "aggregation": "concat",
        "regression_sample_metrics": ["CRPS"],
        "creator": "synthetic_test",
        "reconciliation": None,
    }
    return meta_config
