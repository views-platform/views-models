def get_meta_config():
    """
    Contains the meta data for the model (model algorithm, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    
    meta_config = {
        "name": "nhits_bfc", 
        "algorithm": "NHiTSModel",
        "level": "cm",
        "regression_targets": ["lr_gdp_pcap"],
        "regression_point_metrics": ["MSE", "MSLE"],
        "creator": "Xiaolong",
        "time_steps": 36,
        "evaluation_sequencing": "horizon_chunks",
        "prediction_format": "prediction_frame",
        "rolling_origin_stride": 1,
        "skip_predictions_delivery": True,
    }
    return meta_config
