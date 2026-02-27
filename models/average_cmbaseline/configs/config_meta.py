def get_meta_config():
    """
    Contains the meta data for the model (model algorithm, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """

    meta_config = {
        "name": "average_cmbaseline",
        "algorithm": "AverageModel",
        # Uncomment and modify the following lines as needed for additional metadata:
<<<<<<< HEAD
        "regression_targets": ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"],
        # "queryset": "escwa001_cflong",
        "level": "cm",
        "creator": "Sonja",
        "regression_point_metrics": [
            "RMSLE",
            "MSE",
            "MSLE",
            "y_hat_bar",
            "Pearson",
        ],
=======
        "targets": ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"],
        # "queryset": "escwa001_cflong",
        "level": "cm",
        "creator": "Sonja",
        "metrics": ["RMSLE", "CRPS", "MSE", "MSLE", "y_hat_bar", "MTD", "BCD", "Pearson", "LevelRatio", "KL", "JS", "QuantileLoss", "EMD"],
>>>>>>> origin/chained_scalers_2
    }
    return meta_config
