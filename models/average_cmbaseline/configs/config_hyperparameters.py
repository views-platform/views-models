
def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.
    This configuration is "operational" so modifying these settings will impact the model's behavior during the training.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for training the model, which determine the model's behavior during the training phase.
    """
    
    hyperparameters = {
        'steps': [*range(1, 36 + 1, 1)],
        'time_steps': 36,
        'skip_predictions_delivery': True,
        'n_posterior_samples': 1,
        'regression_targets': ['lr_ged_sb'],
        'window_months': 60,
    }
    return hyperparameters
