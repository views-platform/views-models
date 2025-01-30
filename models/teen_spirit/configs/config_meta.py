def get_meta_config():
    """
    Contains the meta data for the model (model algorithm, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    
    meta_config = {
        "name": "teen_spirit", 
        "algorithm": "RandomForestModel",
        "metrics": ["RMSLE", "CRPS"],
        "depvar": "ln_ged_sb_dep",
        "queryset": "fatalities003_faoprices",
        "level": "cm",
        "creator": "Marina"
    }
    return meta_config
