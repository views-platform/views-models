def get_hp_config():
    """Hyperparameters for short_linen. Operational — changing these changes the forecast.

    short memory: 12 months of observed values to interpolate between.

    ⛔ THIS ARM DIFFERS FROM THE SHARED BASE IN EXACTLY ONE SETTING (window_months). That is the
    design of views-models#695: nine variants against one control, each varying one thing, so a
    difference in score traces to the setting that caused it. Varying the SEED instead would
    give ten copies of one model — the seed is a run-time setting and views-baseline#109
    replicates over it separately.

    `n_samples` is 256, not the 128 of the epic #595 members: views-baseline#109 requires
    every arm to carry the same number of draws, and AP is quantised to S+1 rank levels, so an
    unequal S makes two arms incomparable without raising anything.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for the baseline model.
    """

    hyperparameters = {
        "regression_targets": ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
        "steps": list(range(1, 37)),
        "time_steps": 36,
        "window_months": 12,
        "n_samples": 256,
        "n_posterior_samples": 256,
        "seed": 42,
        "skip_predictions_delivery": True,
    }

    return hyperparameters
