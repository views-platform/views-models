def get_meta_config():
    """
    Contains the metadata for the model (model architecture, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    meta_config = {
        "name": "rude_boy",
        "models": ["cool_cat", "dancing_queen", "elastic_heart", "good_life", "heat_waves", "new_rules", "teenage_dirtbag"],
        "targets": ["lr_ged_sb_dep"],
        "level": "cm", 
        "aggregation": "mean",
        "metrics": ["RMSLE", "CRPS", "MSE", "MSLE", "y_hat_bar"],
        "creator": "Dylan" 
    }
    return meta_config
