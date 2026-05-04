def get_meta_config():
    """
    Contains the metadata for the model (model architecture, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    meta_config = {
        "name": "first_love",
        "models": ["bad_romance", "cold_heart", "free_fallin"], # "revolving_door", "new_rules"
        "regression_targets": ["lr_ged_sb"],
        "level": "cm", 
        "aggregation": "mean",
        "regression_point_metrics": ["RMSLE", "MSE", "MSLE", "y_hat_bar"],
        "regression_sample_metrics": ["CRPS", "y_hat_bar", "twCRPS", "QIS", "MIS", "MCR_sample"],
        "regression_point_baselines": ["average_cmbaseline", "zero_cmbaseline", "locf_cmbaseline"],
        "regression_sample_baselines": ["red_ranger"],
        "creator": "Dylan" 
    }
    return meta_config
