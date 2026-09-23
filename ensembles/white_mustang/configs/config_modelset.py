def get_modelset_config():
    """
    Contains the list of constituent models for the ensemble.

    Returns:
    - modelset_config (dict): A dictionary with the key 'models' listing constituent model names.
    """
    modelset_config = {
        "models": ["lavender_haze", "blank_space"],
    }
    return modelset_config
