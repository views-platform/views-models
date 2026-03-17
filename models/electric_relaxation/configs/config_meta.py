def get_meta_config():
    """
    Contains the common configuration settings for the model (model architecture, name, target variable, level of analysis and deployment status).

    Returns:
    - model_config (dict): A dictionary containing model configuration settings.
    """
    model_config = {
        "name": "electric_relaxation",
        "algorithm": "RandomForestRegressor", 
        "regression_point_metrics": ["RMSLE", "MSE", "MSLE", "y_hat_bar"],
        "regression_targets": ["lr_ged_sb"], 
        "queryset": "escwa001_cflong",
        "level": "cm",
        "creator": "Sara",
        "prediction_format": "dataframe",
        "rolling_origin_stride": 1,
    }
    return model_config 