def get_meta_config():
    """
    Contains the metadata for the model (model architecture, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    meta_config = {
        "name": "chunky_bunny",
        "regression_targets": ["lr_ged_sb"],
        "level": "cm",
        "aggregation": "mean",
        "regression_point_baselines": ["average_cmbaseline", "zero_cmbaseline", "locf_cmbaseline"],
        "regression_point_metrics": ["MSLE", "MSE", "MCR_point", "y_hat_bar"],
        "creator": "Simon",
        "reconciliation": None,
    }
    return meta_config
