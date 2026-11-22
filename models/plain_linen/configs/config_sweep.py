def get_sweep_config():
    """Sweep configuration for plain_linen.

    Pinned to this arm's #695 setting. The point of the ten arms is that each differs from the
    shared base in exactly one place; a sweep that moved several at once would dissolve the
    contrast the set was built to make. The one key this arm varies is present so a sweep CAN
    move it deliberately — views-baseline#109's alpha sweep is an experiment script, not a
    hundred model folders.

    Returns:
    - sweep_config (dict): A dictionary containing the configuration for hyperparameter sweeps.
    """

    sweep_config = {
        'method': 'grid',
        'name': 'plain_linen'
    }

    metric = {
        'name': 'CRPS',
        'goal': 'minimize'
    }
    sweep_config['metric'] = metric

    parameters_dict = {
        'steps': {'value': [*range(1, 36 + 1, 1)]},
        'time_steps': {'value': 36},
        'window_months': {'value': 36},
        'n_samples': {'value': 64},
        'seed': {'value': 42},
    }
    sweep_config['parameters'] = parameters_dict

    return sweep_config
