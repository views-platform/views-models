def get_modelset_config():
    """
    Contains the list of constituent models for the ensemble.

    Returns:
    - modelset_config (dict): A dictionary with the key 'models' listing constituent model names.

    Note:
        chunky_bunny is a clone of the original ``big_chungus`` ensemble (the
        2026-06-04 calibration run): the full 23 constituents — 13 plain
        stepshifters + 6 Hurdle stepshifters + 4 deep-learning models — as opposed
        to ``pink_ponyclub`` which carries only the 19 stepshifter constituents.
    """
    modelset_config = {
        "models": [
            "bittersweet_symphony",
            "brown_cheese",
            "car_radio",
            "counting_stars",
            "demon_days",
            "elastic_heart",
            "fast_car",
            "fluorescent_adolescent",
            "good_riddance",
            "green_squirrel",
            "heavy_rotation",
            "high_hopes",
            "little_lies",
            "national_anthem",
            "new_rules",
            "ominous_ox",
            "plastic_beach",
            "popular_monster",
            "revolving_door",
            "smol_cat",
            "teen_spirit",
            "twin_flame",
            "yellow_submarine",
        ],
    }
    return modelset_config
