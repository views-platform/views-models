def get_meta_config():
    """
    Contains the meta data for the model (model algorithm, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    
    meta_config = {
        "name": "un_fao", 
        "algorithm": "Postprocessor",
        # Uncomment and modify the following lines as needed for additional metadata:
        "targets": ["lr_ged_sb"],
        # "queryset": "escwa001_cflong",
        "level": "pgm",
        "ensemble": "un_fao"
        # "creator": "Your name here",
    }
    return meta_config
