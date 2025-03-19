
def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.
    This configuration is "operational" so modifying these settings will impact the model's behavior during the training.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for training the model, which determine the model's behavior during the training phase.
    """
    
    hyperparameters = {
        'steps': [*range(1, 36 + 1, 1)],
        'submodels_to_train': 1,
        'pred_samples': 10,
        'log_target': False,
        'draw_dist': 'Lognormal',
        'draw_sigma': 0.5,
        'geo_unit_samples': 1.0,
        "parameters": {
            "clf": {
                'n_estimators': 2,
                'n_jobs': -2,
            },
            "reg": {
                'n_estimators': 2,
                'n_jobs': -2,
            }
        }
    }
    return hyperparameters
