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
        'markov_target': 'lr_ged_sb_dep',
        'markov_threshold': 0,
        'markov_method': 'transition',
        'regression_method': 'multi',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': True,
        'rf_class_params': {
            'n_estimators': 500,
        },
        'rf_reg_params': {
            'n_estimators': 500,
            'max_features': 'sqrt',
            'min_samples_split': 2,
            'min_samples_leaf': 5,
        },
    }

    return hyperparameters