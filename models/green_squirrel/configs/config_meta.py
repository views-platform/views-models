def get_meta_config():
    """
    Contains the meta data for the model (model algorithm, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    
    meta_config = {
        "name": "green_squirrel", 
        "algorithm": "HurdleModel",
        "model_clf": "RandomForestClassifier",
        "model_reg": "RandomForestRegressor",
        "metrics": ["RMSLE", "CRPS"],
        "targets": "ln_ged_sb_dep",
        "queryset": "fatalities003_joint_broad",
        "level": "cm",
        "creator": "Borbála"
    }
    return meta_config
