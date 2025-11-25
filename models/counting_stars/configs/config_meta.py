def get_meta_config():
    """
    Contains the meta data for the model (model algorithm, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    
    meta_config = {
        "name": "counting_stars", 
        "algorithm": "XGBRegressor",
        "metrics": ["RMSLE", "CRPS", "MSE", "MSLE", "y_hat_bar"],
        "targets": "lr_ged_sb",
        "queryset": "fatalities003_conflict_history_long",
        "level": "cm",
        "creator": "Borbála"
    }
    return meta_config
