def get_meta_config():
    """
    Contains the meta data for the model (model algorithm, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    
    meta_config = {
        "name": "revolving_door", 
        "algorithm": "NHiTSModel",
        # Uncomment and modify the following lines as needed for additional metadata:
        # "regression_targets": ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"],
        "regression_targets": ["lr_ged_sb"],
        # "queryset": "escwa001_cflong",
        "level": "cm",
        "creator": "Dylan",
        "regression_point_baselines": ["average_cmbaseline", "zero_cmbaseline", "locf_cmbaseline"],
        "regression_point_metrics": ["MCR_point", "MSE", "MSLE", "y_hat_bar"],
        "regression_sample_metrics": ["CRPS", "y_hat_bar"],
        # "regression_sample_baselines": ["red_ranger"],  # commented to match elastic_heart/new_rules/smol_cat; red_ranger's latest wandb run is stale (pre +12mo bump) and trips the report partition check. Does not affect chunky_bunny (point baselines only).
        "rolling_origin_stride": 1,
        "prediction_format": "dataframe",
    }
    return meta_config
