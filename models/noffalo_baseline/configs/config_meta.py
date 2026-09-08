def get_meta_config():
    """
    Contains the meta data for the model (model algorithm, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    
    meta_config = {
        "name": "noffalo_baseline", 
        "algorithm": "AverageModel",
        "level": "cm",
        "regression_targets": ["lr_gdp_pcap"],
        "regression_point_metrics": ["RMSLE", "MSE", "MSLE", "y_hat_bar"],
        "prediction_format": "prediction_frame",
        "rolling_origin_stride": 1,
        "creator": "Borbála",
    }
    return meta_config

