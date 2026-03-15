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
        "name": "purple_alien",
        "algorithm": "HydraNet", 
        "creator": "Simon",
        "level": "pgm",
        
        # ============================================================
        # output format
        # ============================================================

        "prediction_format": "prediction_frame", #"dataframe",
        # "prediction_format": "dataframe",
        "skip_predictions_delivery": True,  # Suspend Track B parquet delivery (OOM mitigation)
      

        # ============================================================
        # diagnostic settings
        # ============================================================
        "diagnostic_visualizations": False, #True,

        # ============================================================
        # evaluation settings 
        # ============================================================
        "regression_sample_metrics": ["twCRPS", "QIS", "MIS", "MCR_sample"],
        "evaluation_profile": "hydranet_ucdp",
        "classification_point_metrics": ["AP"],
        "rolling_origin_stride": 1,
    }
    return meta_config 
