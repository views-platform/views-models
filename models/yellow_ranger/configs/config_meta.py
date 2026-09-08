def get_meta_config():
    """
    Contains the meta data for the model (model algorithm, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """

    meta_config = {
        "name": "yellow_ranger",
        "algorithm": "MixtureBaseline",
        "regression_targets": ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"],
        "level": "cm",
        "creator": "Simon",
        "prediction_format": "prediction_frame",
        "rolling_origin_stride": 1,
        "regression_sample_metrics": ["twCRPS", "QIS", "MIS", "MCR_sample", "CRPS", "y_hat_bar"],
    }
    return meta_config
