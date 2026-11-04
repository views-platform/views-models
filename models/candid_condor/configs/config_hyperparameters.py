def get_hp_config():
    """Hyperparameters for candid_condor. Operational — changing these changes the forecast.

    This member's place in epic #595's table is fixed by `window_months`.
    `n_samples` is 128, the ADR-015 posterior-width standard, and the
    #595 ensembles pool eight members by concatenation to reach 1,024 draws.
    `seed` is declared explicitly because it is a required genome key.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for the baseline model.
    """

    hyperparameters = {
        "regression_targets": ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
        "steps": list(range(1, 37)),
        "time_steps": 36,
        "window_months": 24,
        "n_samples": 128,
        "n_posterior_samples": 128,
        "seed": 42,
        "skip_predictions_delivery": True,
    }

    return hyperparameters
