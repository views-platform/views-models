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
        "targets": ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"],
        # "queryset": "escwa001_cflong",
        "level": "pgm",
        # The forecast ensemble whose .env credentials / upload metadata this
        # postprocessor uses (it does NOT pool the ensemble by target name — it
        # renames its own datafactory actuals via config_queryset.FEATURE_RENAME).
        # rusty_bucket is the FAO global-land forecast ensemble (#77, #143);
        # replaces the ghost orange_ensemble.
        "ensemble": "rusty_bucket"
        # "creator": "Your name here",
    }
    return meta_config
