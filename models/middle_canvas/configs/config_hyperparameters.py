def get_hp_config():
    """Hyperparameters for middle_canvas. Operational — changing these changes the forecast.

    L2: POOLED OVER THE COUNTRY. Plain conflictology, but a cell draws from the last 36 months of every
    cell in its area rather than only its own. Identical to plain_burlap in every key but `pool_by`.

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
        "n_samples": 256,
        "n_posterior_samples": 256,
        "seed": 42,
        "skip_predictions_delivery": True,
        # views-baseline 1.4.0 / #151. REQUIRED and audited. Names the area a cell pools over.
        # NO_AREA = -1: the 76 cells outside FAO GAUL coverage draw from their own history only,
        # exactly as plain_burlap does, and the run logs one WARNING with the count.
        "pool_by": "c_id",
    }

    return hyperparameters
