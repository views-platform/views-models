def get_meta_config():
    """
    Contains the metadata for the model (model architecture, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    meta_config = {
        "name": "gdp_pcap",
        "regression_targets": ["lr_gdp_pcap"],
        "level": "cm", 
        "aggregation": "mean",
        "regression_point_baselines": [],
        "regression_point_metrics": ["MCR_point", "MSE", "MSLE", "y_hat_bar"],
        "creator": "Dylan",
        # "regression_sample_baselines": ["red_ranger"],
        "prediction_format": "prediction_frame",
        "skip_predictions_delivery": True,
    }
    return meta_config
