def get_sweep_config():
    """Sweep configuration for ashen_sparrow.

    Every parameter is pinned to this member's #595 setting. A sweep over a member
    of a reconciled ensemble would break the pairing the ALIGNED_DRAWS mode relies
    on — grid draw i is scaled to country draw i, so member k's settings must stay
    equal to its country twin's. Sweep the ensemble's composition instead.

    Returns:
    - sweep_config (dict): A dictionary containing the configuration for hyperparameter sweeps.
    """

    sweep_config = {
        'method': 'grid',
        'name': 'ashen_sparrow'
    }

    metric = {
        'name': 'CRPS',
        'goal': 'minimize'
    }
    sweep_config['metric'] = metric

    parameters_dict = {
        'steps': {'value': [*range(1, 36 + 1, 1)]},
        'time_steps': {'value': 36},
        'window_months': {'value': 12},
        'n_samples': {'value': 128},
        'seed': {'value': 42},
    }
    sweep_config['parameters'] = parameters_dict

    return sweep_config
