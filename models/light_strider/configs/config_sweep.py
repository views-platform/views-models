def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    This configuration is "operational" so modifying it will change the search strategy, parameter ranges, and other settings for hyperparameter tuning aimed at optimizing model performance.

    Returns:
    - sweep_config (dict): A dictionary containing the configuration for hyperparameter sweeps, defining the methods and parameter ranges used to search for optimal hyperparameters.
    """

    sweep_config = {
        'method': 'grid',
        'name': 'light_strider'
    }

    metric = {
        'name': 'CRPS',
        'goal': 'minimize'
    }
    sweep_config['metric'] = metric

    parameters_dict = {
        'steps': {'value': [*range(1, 36 + 1, 1)]},
        'time_steps': {'value': 36},
        'n_samples': {'value': 64},
        'window_months': {'values': [12, 18, 24, 36, 48, 60]},
        'seed': {'values': [42, 123, 456]},
    }
    sweep_config['parameters'] = parameters_dict

    return sweep_config
