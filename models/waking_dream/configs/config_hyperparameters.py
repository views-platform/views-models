def get_hp_config():
    hyperparameters = {
        'steps': [*range(1, 36 + 1, 1)],
        'time_steps': 36,
        'window_months': 18,
        'lambda_mix': 0.10,
        'n_samples': 64,
    }
    return hyperparameters
