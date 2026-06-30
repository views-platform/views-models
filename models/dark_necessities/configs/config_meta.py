def get_meta_config():
    """
    Contains the meta data for the model (model algorithm, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    
    meta_config = {
        "name": "dark_necessities",
        "algorithm": "TiDEModel",
        # Uncomment and modify the following lines as needed for additional metadata:
        "regression_targets": ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"],
        "level": "pgm",
        "creator": "Dylan",
        "regression_point_baselines": ["average_pgmbaseline", "zero_pgmbaseline", "locf_pgmbaseline"],
        "regression_point_metrics": ["MCR_point", "MSE", "MSLE", "y_hat_bar"],
        # "regression_sample_metrics": ["CRPS", "y_hat_bar", "twCRPS", "QIS", "MIS", "MCR_sample"],
        # "regression_sample_baselines": ["red_ranger"],
        "rolling_origin_stride": 1,
        "prediction_format": "dataframe",
    }
    return meta_config
