def get_meta_config():
    """
    Contains the metadata for the model (model architecture, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    meta_config = {
        "name": "less_hope",
        "models": ["average_pgmbaseline", "zero_pgmbaseline"],
        "targets": ["lr_ged_sb"],
        "level": "pgm",
        "aggregation": "mean",
        "metrics": [
            "RMSLE",
            "MSE",
            "MSLE",
            "y_hat_bar",
        ],
        "creator": "Sonja",
    }
    return meta_config
