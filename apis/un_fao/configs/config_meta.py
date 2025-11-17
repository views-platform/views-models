def get_meta_config():
    """
    Contains the meta data for the model (model algorithm, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    
    meta_config = {
        "name": "un_fao", 
        "algorithm": "API",
        # "creator": "Your name here",
        "historical_targets": ["lr_ged_sb"],
        "ensemble": "orange_ensemble",
        "level": "pgm"
    }
    return meta_config
