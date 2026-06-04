def get_meta_config():
    meta_config = {
        "name": "stellar_horizon",
        "models": ["bright_starship", "bold_comet", "blazing_meteor"],
        "regression_targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
        "level": "pgm",
        "aggregation": "concat",
        "regression_sample_metrics": ["CRPS", "QS_sample", "MCR_sample"],
        "evaluation_profile": "hydranet_ucdp",
        "creator": "Simon",
        "reconciliation": None,
    }
    return meta_config
