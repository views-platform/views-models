def get_meta_config():
    """
    Contains the meta data for the preprocessor.
    This config is for documentation purposes only, and modifying it will not affect the preprocessor, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing preprocessor meta configuration.
    """
    
    meta_config = {
        "name": "active_learner", 
        "algorithm": "ActiveLearner",
        "creator": "Dylan, Xiaolong, Sonja, Simon"
    }
    return meta_config
