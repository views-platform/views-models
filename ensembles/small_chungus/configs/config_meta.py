def get_meta_config():
    """
    Contains the metadata for the model (model architecture, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    meta_config = {
        "name": "small_chungus",
        "models": [
            "dark_necessities",
            "dancing_monkey"
            ],
        "regression_targets": ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"],
        "level": "pgm", 
        "aggregation": "mean",
        "regression_point_baselines": ["average_pgmbaseline", "zero_pgmbaseline", "locf_pgmbaseline"],
        "regression_point_metrics": ["MCR_point", "MSE", "MSLE", "y_hat_bar"],
        "creator": "Dylan",
        "rolling_origin_stride": 1,
        "prediction_format": "prediction_frame",
        "skip_predictions_delivery": True,
    }
    return meta_config
