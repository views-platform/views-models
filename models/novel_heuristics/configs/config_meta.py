def get_meta_config():
    """
    Contains the meta data for the model (model algorithm, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    
    meta_config = {
        "name": "novel_heuristics", 
        "algorithm": "NBEATSModel",
        "regression_targets": ["lr_ged_sb_dep"],
        # "queryset": "escwa001_cflong",
        "level": "cm",
        "creator": "Simon",
        "regression_point_metrics": ["RMSLE", "MSE", "MSLE", "y_hat_bar"],
    }
    return meta_config
