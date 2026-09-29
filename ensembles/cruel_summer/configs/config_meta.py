def get_meta_config():
    """
    Contains the metadata for the model (model architecture, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    meta_config = {
        "name": "cruel_summer",
        "regression_point_baselines": ["average_cmbaseline", "zero_cmbaseline", "locf_cmbaseline"],
        "regression_point_metrics": ["RMSLE", "MSE", "MSLE", "y_hat_bar"],
        "regression_targets": ["lr_ged_sb"], 
        "level": "cm", 
        "aggregation": "median", 
        "creator": "Xiaolong",
        "reconciliation": None
    }
    return meta_config
