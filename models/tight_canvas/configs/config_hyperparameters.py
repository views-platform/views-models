def get_hp_config():
    """Hyperparameters for tight_canvas. Operational — changing these changes the forecast.

    THE NEIGHBOURHOOD CONTROL. Plain conflictology, but a cell draws from the last 36 months of
    every cell within 5 grid cells of it, itself included, every cell and month weighted the
    same. Identical to `plain_burlap` in every key but `radius`, which `plain_burlap` does not
    have at all.

    ⛔ THIS ARM DIFFERS FROM THE SHARED BASE IN EXACTLY ONE SETTING. That is the design of
    views-models#695: variants against one control, each varying one thing, so a difference in
    score traces to the setting that caused it.

    ⛔ WHY `radius` IS 5 AND MUST NOT BE TUNED. It is not a free parameter. It is pinned to
    `NEAR_RADIUS` in views-baseline's `matryoshka/geometry.py`, which is what the `near` doll of
    the eight nested-doll models uses (ADR-024, amendment 2026-11-06). The point of this model is
    to be that doll's single-level control, so a different radius would make it a control for
    something no other model in the comparison contains. If `NEAR_RADIUS` ever moves, this moves
    with it.

    The disc is straight-line grid distance, d <= 5, which is 80 neighbours plus the cell: about
    81 cells, not a square. `radius` must be a whole number; views-baseline rejects a float, and
    a rounded 1.4142135 would silently select 4 cells instead of 8 at radius "root 2".

    ⚠️ FOR THE PAPER, measured by views-baseline on the real 64,818 `land` cells: 8 of them have
    no other land cell within 5 cells. For those 8 this model IS plain conflictology, because the
    disc holds only the cell itself. That is correct behaviour, not a defect, but it means the
    neighbourhood arm and the cell arm are identical on 8 cells and the paper should say so
    rather than let a reader assume all 64,818 differ.

    ⛔ S = 64, IN THREE PLACES AT ONCE — here, `n_posterior_samples` below, and
    `config_sweep.py` (C-104, C-85). CRPS and the rank metrics are quantised to S+1 levels, so a
    model at 256 cannot be compared with one at 64, and nothing raises when they are. The twelve
    archives of the 2026-11-13 run and the four of 2026-11-17 all carry 64 draws, verified from
    the arrays themselves (2,333,448 x 64; see docs/matryoshka_experiment_catalog.md).

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
        # views-baseline 1.6.0 / #178. REQUIRED and audited; a whole number in 0..359. Pinned to
        # NEAR_RADIUS — see the note above on why this is not a tunable.
        "radius": 5,
    }

    return hyperparameters
