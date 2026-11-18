def get_meta_config():
    """
    Contains the meta data for the model (model algorithm, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    
    meta_config = {
        "name": "sandy_valley", 
        "algorithm": "SkDistributionalHurdleModel",
        "model_clf": "LGBMClassifier",
        "model_reg": "TabICLQuantileRegressor",
        "regression_sample_metrics": ["CRPS", "y_hat_bar", "twCRPS", "QIS", "MIS", "MCR_sample"],
        "regression_targets": ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"],
        "level": "pgm",
        "creator": "Daniel Mittermaier",
        "prediction_format": "prediction_frame",
        "rolling_origin_stride": 1,
    }
    return meta_config