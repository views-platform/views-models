def get_modelset_config():
    """
    Contains the list of constituent models for the ensemble.

    Returns:
    - modelset_config (dict): A dictionary with the key 'models' listing constituent model names.
    """
    modelset_config = {
        "models": ["smol_cat", "dancing_queen", "elastic_heart", "new_rules"],
    }
    return modelset_config
