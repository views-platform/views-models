def get_modelset_config():
    """
    Contains the list of constituent models for the ensemble.

    Returns:
    - modelset_config (dict): A dictionary with the key 'models' listing constituent model names.
    """
    modelset_config = {
        "models": [
            "frozen_crown",
            "silver_drift",
        ],
    }
    # modelset_config = {
    #     "models": [
    #         "smol_cat",
    #         "new_rules",
    #         "revolving_door",
    #         "elastic_heart",
    #         "crimson_tide",
    #         "rapid_fire",
    #         "wild_storm",
    #         "iron_will",
    #     ],
    # }
    return modelset_config
