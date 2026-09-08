def get_meta_config():
    """
    Contains the metadata for the model (model architecture, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    meta_config = {
        "name": "rude_boy",
        "models": [
            "bittersweet_symphony",
            "bright_star",
            "brown_cheese",
            # "bus_radio",
            "car_radio",
            "counting_stars",
            "crimson_tide",
            "demon_days",
            # "demon_weeks",
            "elastic_heart",
            "fast_car",
            "fluorescent_adolescent",
            # "fluorescent_adult",
            "frozen_peak",
            "good_riddance",
            "green_squirrel",
            "heavy_rotation",
            "high_hopes",
            # "higher_hopes",
            "iron_will",
            "little_lies",
            "national_anthem",
            "new_rules",
            "ominous_ox",
            "plastic_beach",
            "popular_monster",
            "rapid_fire",
            "revolving_door",
            "shadow_wolf",
            "smol_cat",
            "swift_current",
            "teen_spirit",
            "twin_flame",
            # "unpopular_monster",
            "wild_storm",
            "yellow_submarine",
        ],
        "regression_targets": ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"],
        "level": "cm", 
        "aggregation": "mean",
        "regression_point_baselines": ["average_cmbaseline", "zero_cmbaseline", "locf_cmbaseline"],
        "regression_point_metrics": ["MCR_point", "MSE", "MSLE", "y_hat_bar"],
        "creator": "Dylan",
        # "regression_sample_baselines": ["red_ranger"],
    }
    return meta_config
