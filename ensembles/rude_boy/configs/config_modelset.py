def get_modelset_config():
    """
    Contains the list of constituent models for the ensemble.

    Returns:
    - modelset_config (dict): A dictionary with the key 'models' listing constituent model names.
    """
    modelset_config = {
        "models": ['bright_star', 
                #    'car_radio', 
                   'bus_radio',
                   'crimson_tide', 
                #    'demon_days', 
                   'demon_weeks', 
                   'frozen_peak', 
                   'iron_will', 
                   'rapid_fire', 
                   'shadow_wolf', 
                   'smol_cat', 
                   'swift_current', 
                #    'popular_monster', 
                   'unpopular_monster', 
                #    'teen_spirit',
                   'adult_spirit'
                   ],
    }
    return modelset_config
