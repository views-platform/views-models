def get_meta_config():
    """
    Contains the meta data for the model (model algorithm, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    
    meta_config = {
        "name": "vdem_extractor", 
        "algorithm": "VDEMData",
        # Uncomment and modify the following lines as needed for additional metadata:
        # "targets": ["ln_ged_sb_dep"],
        # "queryset": "escwa001_cflong",
        # "level": "pgm",
        # "creator": "Your name here",
    }
    return meta_config
