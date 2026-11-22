def get_hp_config():
    """Hyperparameters for plain_linen. Operational — changing these changes the forecast.

    the base window, 36 months — the shared setting every other arm starts from.

    ⛔ THIS ARM DIFFERS FROM THE SHARED BASE IN EXACTLY ONE SETTING. That is the
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
        "window_months": 36,
        # S=64: what #695 actually ran (views-baseline C-54), and matched to the #728
        # comparison. Move all three together (C-104/C-85) — here, n_posterior_samples
        # below, and config_sweep.py. The config previously said 256, which this arm
        # never ran at: 256 needs ~192 GB peak disk against a 100 GB pod quota (#717).
        "n_samples": 64,
        "n_posterior_samples": 64,
        "seed": 42,
        "skip_predictions_delivery": True,
    }

    return hyperparameters
