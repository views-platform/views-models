
def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.
    This configuration is "operational" so modifying these settings will impact the model's behavior during the training.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for training the model, which determine the model's behavior during the training phase.
    """
    
    hyperparameters = {
        'steps': [*range(1, 36 + 1, 1)],
        'submodels_to_train': 5,
        'max_features': 0.3,
        'max_depth': None,
        'max_samples': None,
        'pred_samples': 10,
        'draw_dist': '',
        'draw_sigma': 0.3,
        'geo_unit_samples': 1.0,
        'n_jobs': -2,
        "parameters": {
            "clf": {
                'n_estimators': 20,
            },
            "reg": {
                'n_estimators': 20,
            }
        }
    }
    return hyperparameters
