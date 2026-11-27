def get_modelset_config():
    """
    Contains the list of constituent models for the ensemble.

    Returns:
    - modelset_config (dict): A dictionary with the key 'models' listing constituent model names.
    """
    modelset_config = {
        "models": ["lucid_dream", "vivid_dream", "waking_dream"],
    }
    return modelset_config
