def get_meta_config():
    """
    Contains the metadata for the model (model architecture, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    meta_config = {
        "name": "first_love",
        "models": ["bad_romance", "free_fallin", "cold_heart", "beautiful_people", "holy_grail"],
        "regression_targets": ["lr_ged_sb"],
        "level": "cm", 
        "aggregation": "concat",
        # "regression_point_baselines": ["average_cmbaseline", "zero_cmbaseline", "locf_cmbaseline"],
        # "regression_point_metrics": ["RMSLE", "MSE", "MSLE", "y_hat_bar"],
        "regression_sample_metrics": ["y_hat_bar", "twCRPS", "QIS", "MIS", "MCR_sample"],
        # "regression_point_baselines": ["average_cmbaseline", "zero_cmbaseline", "locf_cmbaseline"],
        "regression_sample_baselines": ["red_ranger"],
        "creator": "Dylan" 
    }
    return meta_config
