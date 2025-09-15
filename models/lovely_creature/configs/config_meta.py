def get_meta_config():
    """
    Contains the meta data for the model (model algorithm, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    
    meta_config = {
        "name": "lovely_creature", 
        "algorithm": "ShurfModel",
        "targets": ["sb_best"],
        "level": "cm",
        "creator": "Håvard",
        "model_reg": "XGBRegressor",
        "model_clf": "XGBClassifier",
        "metrics": ["RMSLE", "CRPS", "MSE", "MSLE", "y_hat_bar"],
        "queryset": "uncertainty_broad_nolog",
    }
    return meta_config
