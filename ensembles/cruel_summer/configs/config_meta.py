def get_meta_config():
    """
    Contains the metadata for the model (model architecture, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    meta_config = {
        "name": "cruel_summer",
        "models": ["bittersweet_symphony", "brown_cheese"],
        "metrics": ["RMSLE", "CRPS", "MSE", "MSLE", "y_hat_bar"],
        "targets": "lr_ged_sb", 
        "level": "cm", 
        "aggregation": "median", 
        "creator": "Xiaolong",
        "reconciliation": None,
        "use_weights": False,
        "weights": {  # Only used if use_weights is True
            "bittersweet_symphony": 0.5,
            "brown_cheese": 0.5
        }
    }
    return meta_config
