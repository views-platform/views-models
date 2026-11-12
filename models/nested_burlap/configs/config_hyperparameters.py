def get_hp_config():
    """Hyperparameters for nested_burlap. Operational — changing these changes the forecast.

    THE FULL MODEL (#717). Matryoshka draws each cell from nested groups of cells; `dolls` names which.
    Identical to nested_burlap in every key but `dolls`, so a difference traces to the levels used.
    ⛔ NO `window_months`: Matryoshka has no window — its history is all the data passed, and the
    key is absent from its genome, so leaving it in would be a silent unaudited parameter.

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
        "n_samples": 256,
        "n_posterior_samples": 256,
        "seed": 42,
        "skip_predictions_delivery": True,
        # ADR-024 / views-baseline#153. All six are REQUIRED and audited by
        # ReproducibilityGate.ALGORITHM_GENOMES["MatryoshkaConflictology"]; a missing key or a
        # wrong type fails at config validation, not deep in a run.
        "country_column": "c_id",      # the queryset renames gaul0_code -> c_id
        "backtest_cutoffs": 4,
        "backtest_horizon": 36,
        "stability_refits": False,     # #148: every stage refits; True costs ~5x and changes no forecast
        "dolls": ["cell", "near", "country", "map"],
    }

    return hyperparameters
