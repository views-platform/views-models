def get_hp_config():
    """Hyperparameters for rolling_burlap. Operational — changing these changes the forecast.

    THE MOVING-WINDOW CONTROL at the cell level (views-models#716). Every value here is
    `plain_burlap`'s; the only difference between the two models is `algorithm`, in
    `config_meta.py`. That is the whole design: if a single setting differed, a score gap
    between them would no longer measure the information rule alone.

    ⛔ S = 64, IN THREE PLACES AT ONCE — here, `n_posterior_samples` below, and
    `config_sweep.py` (C-104, C-85). 64 and not the 256 this file carried when it was written
    on 2026-11-09: the twelve models of the 2026-11-13 run all produced 64 draws, VERIFIED from
    the arrays themselves (2,333,448 x 64; see docs/matryoshka_experiment_catalog.md). CRPS and
    the rank metrics are quantised to S+1 levels, so a model at 256 cannot be compared with one
    at 64 — and nothing raises when they are. The 256 came from reading views-baseline#109's
    equal-S requirement and then picking the wrong value to equalise on.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for the baseline model.
    """

    hyperparameters = {
        "regression_targets": ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
        "steps": list(range(1, 37)),
        "time_steps": 36,
        "window_months": 36,
        "n_samples": 64,
        "n_posterior_samples": 64,
        "seed": 42,
        "skip_predictions_delivery": True,
    }

    return hyperparameters
