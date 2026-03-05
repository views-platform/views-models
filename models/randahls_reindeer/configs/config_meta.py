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
        "targets": ["ln_ged_sb_dep"],
        "queryset": "fatalities002_joint_narrow",
        "level": "cm",
        "creator": "Luuk Boekestein",
        "metrics": ["RMSLE", "CRPS", "MSE", "MSLE", "y_hat_bar"],
    }
    return meta_config
