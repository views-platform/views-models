def get_meta_config():
    """
    Contains the metadata for the model (model architecture, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    meta_config = {
        "name": "big_chungus",
        "models": [
            "little_talks",
            "mister_bluesky"
            ],
        "regression_targets": ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"],
        "level": "pgm", 
        "aggregation": "concat",
        "creator": "Dylan",
        "regression_sample_metrics": ["y_hat_bar", "twCRPS", "QIS", "MIS", "MCR_sample", "CRPS"],
        "regression_sample_baselines": ["black_ranger", "blue_ranger", "pink_ranger", "white_ranger"],
        "rolling_origin_stride": 1,
        "prediction_format": "prediction_frame",
        "skip_predictions_delivery": True,
        "reconciliation": "pgm_cm_point",
        "reconcile_with": "first_love",
    }
    return meta_config
