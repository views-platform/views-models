def get_meta_config():
    """
    Contains the meta data for the model (model algorithm, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    
    meta_config = {
        "name": "purple_haze", 
        "algorithm": "SHURF",
        # Uncomment and modify the following lines as needed for additional metadata:
        "depvar": "ged_sb_dep",
        # "queryset": "escwa001_cflong",
        "level": "cm",
        "creator": "Håvard",
        "model_reg": "RandomForestModel",
        "model_clf": "RandomForestModel",
        "metrics": ["RMSLE", "CRPS"],
    }
    return meta_config
