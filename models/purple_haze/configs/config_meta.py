def get_meta_config():
    """
    Contains the meta data for the model (model algorithm, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    
    meta_config = {
        "name": "purple_haze", 
        "algorithm": "ShurfModel",
        "targets": ["sb_best"],
        "level": "cm",
        "creator": "Håvard",
        "model_reg": "RandomForestRegressor",
        "model_clf": "RandomForestClassifier",
        "metrics": ["RMSLE", "CRPS", "MSE"],
        "queryset": "uncertainty_broad_nolog",
    }
    return meta_config
