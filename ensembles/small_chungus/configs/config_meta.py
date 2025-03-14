def get_meta_config():
    """
    Contains the metadata for the model (model architecture, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    meta_config = {
        "name": "small_chungus", # Eg. "happy_kitten"
        "models": ["bittersweet_symphony",
                    "brown_cheese",
                    "car_radio",
                    "counting_stars",
                    "demon_days",
                    # "electric_relaxation",
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
                    "yellow_submarine"], # Eg. ["model1", "model2", "model3"]
        "depvar": "ln_ged_sb_dep",  # Eg. "ln_ged_sb_dep"
        "level": "cm", # Eg. "pgm", "cm"
        "aggregation": "mean", # Eg. "median", "mean"
        "creator": "Your name here",
        "metrics": ["RMSLE", "CRPS"],
    }
    return meta_config
