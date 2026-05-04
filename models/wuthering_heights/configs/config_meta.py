def get_meta_config():
    """
    Contains the meta data for the model (model algorithm, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    
    meta_config = {
        "name": "wuthering_heights", 
        "algorithm": "ShurfModel",
        "regression_targets": ["lr_sb_best"],
        "level": "cm",
        "creator": "Håvard",
        "prediction_format": "dataframe",
        "model_reg": "XGBRegressor",
        "model_clf": "XGBClassifier",
        "regression_point_metrics": ["RMSLE", "MSE", "MSLE", "y_hat_bar"],
        "queryset": "uncertainty_deep_conflict_nolog",
        "rolling_origin_stride": 1,
    }
    return meta_config
