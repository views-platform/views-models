def get_meta_config():
    """
    Contains the meta data for the model (model algorithm, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    
    meta_config = {
        "name": "smol_cat", 
        "algorithm": "TiDEModel",
        # Uncomment and modify the following lines as needed for additional metadata:
        "targets": ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"],
        # "queryset": "escwa001_cflong",
        "level": "cm",
        "creator": "Dylan",
        "metrics": ["RMSLE", "CRPS", "MSE", "MSLE", "y_hat_bar", "MTD", "BCD", "Pearson", "LevelRatio"],
    }
    return meta_config
