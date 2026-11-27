def get_hp_config():
    """Hyperparameters for plain_burlap. Operational — changing these changes the forecast.

    THE CONTROL. The base config and nothing else, so the nine have a like-for-like reference.

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
        # ⛔ S 256 -> 64 on 2026-11-13 to match the #728 comparison and what #695 ACTUALLY RAN.
        # views-baseline's call (they designed #731; plain_burlap is its L0 control): an L0 at 256
        # against L1/L2/L3 at 64 is not a comparison, because AP is quantised to S+1 rank levels.
        # For #695 the 64 was a disk workaround, not a considered choice — the plan was 256, the
        # first land run at 256 was killed at 87.3 GB against an 87.5 GiB limit during evaluation,
        # #121 made 256 fit in MEMORY, and C-54 then records disk as the binding limit at ~180 GB
        # written per model. All ten #695 arms ran at 64 and exited 0; none of their configs said so.
        # ⚠️ Three places, always together: here, n_posterior_samples below, config_sweep.py (C-104, C-85).
        # The other nine #695 arms are deliberately NOT changed — they are not part of #728.
        "n_samples": 64,
        "n_posterior_samples": 64,
        "seed": 42,
        "skip_predictions_delivery": True,
    }

    return hyperparameters
