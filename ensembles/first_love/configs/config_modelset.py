def get_modelset_config():
    """
    Contains the list of constituent models for the ensemble.

    Returns:
    - modelset_config (dict): A dictionary with the key 'models' listing constituent model names.
    """
    modelset_config = {
        "models": ["bad_romance", 
                   "free_fallin", 
                   "cold_heart", 
                   "beautiful_people", 
                #    "holy_grail"
                   ],
    }
    return modelset_config
