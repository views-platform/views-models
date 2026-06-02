def get_hp_config():
    hyperparameters = {
        'steps': [*range(1, 36 + 1, 1)],
        'time_steps': 36,
        'window_months': 18,
        'lambda_mix': 0.05,
        'n_samples': 64,
        'n_posterior_samples': 64,
        'regression_targets': ['synth_target'],
        'skip_predictions_delivery': True,
    }
    return hyperparameters
