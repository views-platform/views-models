def get_meta_config():
    """
    Contains the metadata for the model (model architecture, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    meta_config = {
        "name": "pink_ponyclub",
        "models": [
            "bittersweet_symphony",
            "brown_cheese",
            "car_radio",
            "counting_stars",
            "demon_days",
            "fast_car",
            "fluorescent_adolescent",
            "good_riddance",
            "green_squirrel",
            "heavy_rotation",
            "high_hopes",
            "little_lies",
            "national_anthem",
            "ominous_ox",
            "plastic_beach",
            "popular_monster",
            "teen_spirit",
            "twin_flame",
            "yellow_submarine",
        ],
        "targets": "lr_ged_sb_dep",
        "level": "cm",
        "aggregation": "mean",
        "metrics": ["RMSLE", "CRPS", "MSE", "MSLE", "y_hat_bar"],
        "creator": "Xiaolong",
        "reconciliation": None,
    }
    return meta_config
