def get_meta_config():
    """
    Contains the meta data for the model (model algorithm, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    
    meta_config = {
        "name": "summon_nostradamus", 
        "algorithm": "XGBRegressor",
        "targets": "ln_ged_sb_dep",
        "level": "cm",
        "queryset": "qs_nowcasting_cm",
        "metrics": ["RMSLE", "CRPS"],
        "creator": "Chandler
    }
    return meta_config
