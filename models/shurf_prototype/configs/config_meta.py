def get_meta_config():
    """
    Contains the meta data for the model (model algorithm, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    
    meta_config = {
        "name": "shurf_prototype", 
        "algorithm": "SHURF",
        # Uncomment and modify the following lines as needed for additional metadata:
        "depvar": "ged_sb_dep",
        # "queryset": "escwa001_cflong",
        "level": "cm",
        "creator": "Meow",
        "model_reg": "RandomForestModel",
        "model_clf": "RandomForestModel",
    }
    return meta_config
