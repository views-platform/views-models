def get_meta_config():
    """
    Contains the common configuration settings for the model (model architecture, name, target variable, level of analysis and deployment status).

    Returns:
    - model_config (dict): A dictionary containing model configuration settings.
    """
    model_config = {
        "name": "electric_relaxation",
        "algorithm": "RandomForestRegressor", 
        "metrics": ["RMSLE", "CRPS", "MSE", "MSLE", "y_hat_bar"],
        "targets": "ged_sb_dep", 
        "queryset": "escwa001_cflong",
        "level": "cm",
        "creator": "Sara" 
    }
    return model_config 