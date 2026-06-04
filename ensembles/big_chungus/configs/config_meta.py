def get_meta_config():
    """
    Contains the metadata for the model (model architecture, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    meta_config = {
        "name": "big_chungus",
        "models": [
            "smol_cat", 
            "revolving_door", 
            "elastic_heart", 
            "new_rules", 
            # "good_life",
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
            "yellow_submarine",],
        "regression_targets": ["lr_ged_sb"],
        "level": "cm", 
        "aggregation": "mean",
        "regression_point_baselines": ["average_cmbaseline", "zero_cmbaseline", "locf_cmbaseline"],
        "regression_point_metrics": ["RMSLE", "MSE", "MSLE", "y_hat_bar"],
        "creator": "Dylan",
        # "regression_sample_baselines": ["red_ranger"],
    }
    return meta_config
