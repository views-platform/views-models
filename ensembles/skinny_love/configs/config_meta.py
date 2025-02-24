def get_meta_config():
    """
    Contains the metadata for the model (model architecture, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    meta_config = {
        "name": "skinny_love",
        "models": [
            "bad_blood",
            "blank_space",
            "caring_fish",
            "chunky_cat",
            "dark_paradise",
            "invisible_string",
            "lavender_haze",
            "midnight_rain",
            "old_money",
            "orange_pasta",
            "wildest_dream",
            "yellow_pikachu",
        ],
        "depvar": "ln_ged_sb_dep",
        "level": "pgm",
        "aggregation": "mean",
        "metrics": ["RMSLE", "CRPS"],
        "creator": "Xiaolong",
    }
    return meta_config
