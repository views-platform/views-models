def get_meta_config():
    """
    Contains the meta data for the model (model algorithm, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """

    meta_config = {
        "name": "light_strider",
        "algorithm": "ConflictologyModel",
        "creator": "Simon",
        "level": "pgm",
        "prediction_format": "prediction_frame",
        "regression_sample_metrics": ["CRPS", "QS_sample", "MCR_sample", "Brier_rgs_sample"],
        "evaluation_profile": "hydranet_ucdp",
        "rolling_origin_stride": 1,
    }
    return meta_config
