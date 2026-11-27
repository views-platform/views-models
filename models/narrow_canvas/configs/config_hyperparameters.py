def get_hp_config():
    """Hyperparameters for narrow_canvas. Operational — changing these changes the forecast.

    L1: POOLED OVER THE ADMIN-1 UNIT. Plain conflictology, but a cell draws from the last 36 months of every
    cell in its area rather than only its own. Identical to plain_burlap in every key but `pool_by`.

    ⛔ THIS ARM DIFFERS FROM THE SHARED BASE IN EXACTLY ONE SETTING. That is the
    design of views-models#695: nine variants against one control, each varying one thing, so a
    difference in score traces to the setting that caused it. Varying the SEED instead would
    give ten copies of one model — the seed is a run-time setting and views-baseline#109
    replicates over it separately.

    `n_samples` is 64 (see the note below), not the 128 of the epic #595 members: views-baseline#109 requires
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
        # ⛔ S=64, NOT 256, and the number is DECLARED here rather than applied by hand on a
        # pod. The #695 campaign ran at 64 via an edit made on the pod, which died with it —
        # so no run could be reproduced or audited from its config. 256 needs ~192 GB peak
        # disk against a 100 GB pod quota (#717); views-baseline measured ~180 GB written per
        # model at 256 and 7.17 GB per origin. A full-256 run waits on a disk solution.
        # ⚠️ S must be IDENTICAL across every source in the comparison and must move in all
        # THREE places at once — here, n_posterior_samples below, and config_sweep.py
        # (C-104, C-85). Changing one is how two arms silently stop being comparable.
        "n_samples": 64,
        "n_posterior_samples": 64,
        "seed": 42,
        "skip_predictions_delivery": True,
        # views-baseline 1.4.0 / #151. REQUIRED and audited. Names the area a cell pools over.
        # NO_AREA = -1: the 76 cells outside FAO GAUL coverage draw from their own history only,
        # exactly as plain_burlap does, and the run logs one WARNING with the count.
        "pool_by": "gaul1_code",
    }

    return hyperparameters
