def get_meta_config():
    """
    Contains the meta data for the model (model architecture, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    meta_config = {
        "name": "old_money",
        "algorithm": "HurdleModel",
        "model_clf": "LGBMClassifier",
        "model_reg": "LGBMRegressor",
        "metrics": ["MSE", "MSLE", "y_hat_bar"],
        "targets": "ln_ged_sb_dep", 
        "queryset": "fatalities003_pgm_escwa_drought",
        "level": "pgm",
        "creator": "Xiaolong"
    }
    return meta_config