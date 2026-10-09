def get_meta_config():
    """
    Contains the meta data for the model (model algorithm, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    
    meta_config = {
        "name": "good_riddance", 
        "algorithm": "XGBRFRegressor",
        "regression_point_baselines": ["average_cmbaseline", "zero_cmbaseline", "locf_cmbaseline"],
        "regression_point_metrics": ["MSLE", "MSE", "MCR_point", "y_hat_bar"],
        "regression_targets": ["lr_ged_sb"],
        "queryset": "fatalities003_joint_narrow",
        "level": "cm",
        "creator": "Marina",
        "prediction_format": "dataframe",
        "rolling_origin_stride": 1,
    }
    return meta_config
