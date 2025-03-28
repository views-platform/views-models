def get_meta_config():
    """
    Contains the meta data for the model (model architecture, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    meta_config = {
        "name": "wildest_dream",
        "algorithm": "HurdleModel",
        "model_clf": "XGBClassifier",
        "model_reg": "XGBRegressor",
        "metrics": ["RMSLE", "CRPS"],
        "targets": "ln_ged_sb_dep", 
        "queryset": "fatalities003_pgm_conflict_sptime_dist",
        "level": "pgm",
        "creator": "Xiaolong"
    }
    return meta_config