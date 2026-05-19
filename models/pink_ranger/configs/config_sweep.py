
def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    This configuration is "operational" so modifying it will change the search strategy, parameter ranges, and other settings for hyperparameter tuning aimed at optimizing model performance.

    Returns:
    - sweep_config (dict): A dictionary containing the configuration for hyperparameter sweeps, defining the methods and parameter ranges used to search for optimal hyperparameters.
    """

    sweep_config = {
        'method': 'grid',
        'name': 'pink_ranger'
    }

    metric = {
        'name': 'MSE',
        'goal': 'minimize'
    }
    sweep_config['metric'] = metric

    parameters_dict = {
        'steps': {'value': [*range(1, 36 + 1, 1)]},
        'time_steps': {'value': 36},
        'n_samples': {'value': 256},
        'lambda_mix': {'values': [0.0, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]},
        'window_months': {'values': [6, 12, 18, 24, 36, 48, 60]},
    }
    sweep_config['parameters'] = parameters_dict

    return sweep_config
