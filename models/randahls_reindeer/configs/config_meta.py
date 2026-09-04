def get_meta_config():
    """
    Contains the meta data for the model (model algorithm, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    
    meta_config = {
        "name": "randahls_reindeer", 
        "algorithm": "MarkovModel",
        "regression_point_metrics": ["MSE", "MSLE", "y_hat_bar", "MCR_point"],
        "regression_targets": ["lr_ged_sb"],
        "queryset": "markov_joint_narrow",
        "level": "cm",
        "creator": "Luuk Boekestein",
        "prediction_format": "prediction_frame",
        "rolling_origin_stride": 1,
        "skip_predictions_delivery": True,
    }
    return meta_config