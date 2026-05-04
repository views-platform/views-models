def get_meta_config():
    """
    Contains the meta data for the model (model algorithm, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """

    meta_config = {
        "name": "zero_pgmbaseline",
        "algorithm": "ZeroModel",
        "regression_targets": ["lr_ged_sb"],
        "level": "pgm",
        "creator": "Sonja",
        "prediction_format": "dataframe",
        "rolling_origin_stride": 1,
        "regression_point_metrics": ["MSE", "MSLE"],
    }
    return meta_config
