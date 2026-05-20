def get_meta_config():
    """
    Contains the meta data for the model (model architecture, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    meta_config = {
        # ============================================================
        # General information
        # ============================================================
        "name": "blue_stranger",
        "algorithm": "HydraNet", 
        "creator": "Simon",
        "level": "pgm",
        
        # ============================================================
        # output format
        # ============================================================

        "prediction_format": "prediction_frame", #"dataframe",
        # "prediction_format": "dataframe",
        # ============================================================
        # diagnostic settings
        # ============================================================
        "diagnostic_visualizations": False, #True,

        # ============================================================
        # evaluation settings 
        # ============================================================
        "regression_sample_metrics": ["CRPS", "QS_sample", "MCR_sample"],
        "classification_sample_metrics": ["Brier_cls_sample"],
        "evaluation_profile": "hydranet_ucdp",
        "rolling_origin_stride": 1,
    }
    return meta_config 
