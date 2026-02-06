def get_meta_config():
    """
    Contains the meta data for the model (model algorithm, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    
    meta_config = {
        "name": "temporal_disaggregation", 
        "algorithm": "TSMixerModel",
        "level": "cm",
        "targets": ["lr_gdp_pcap"],
        "metrics": ["RMSLE", "CRPS", "MSE", "MSLE", "y_hat_bar"],
        "creator": "Xiaolong",
    }
    return meta_config
