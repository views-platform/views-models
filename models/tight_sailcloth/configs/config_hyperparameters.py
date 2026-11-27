def get_hp_config():
    """Hyperparameters for tight_sailcloth. Operational — changing these changes the forecast.

    THE MOVING-WINDOW NEIGHBOURHOOD CONTROL (views-baseline#178). Every value here is
    `tight_canvas`'s; the only difference between the two models is `algorithm`, in
    `config_meta.py`. That is the whole design: if a single setting differed, a score gap between
    them would no longer measure the information rule alone.

    The two differ only in WHEN the window ends. `tight_canvas` draws from the last 36 months up
    to `train_end` at every origin; this model's window ends at each rolling origin, so it sees
    more data as the origin advances.

    ⛔ AT ORIGIN 0 THIS MODEL MUST EQUAL `tight_canvas` BYTE FOR BYTE. At the first origin the
    moving window has seen exactly what the fixed window has seen, so the predictions must be
    EQUAL, not close. That is the acceptance check for this model and it is the only check it
    has; `tools/audit/verify_experiment_data.py` declares the pair and reports it on every run.
    `tight_canvas` has no check of its own and must therefore run FIRST.

    ⛔ WHY `radius` IS 5 AND MUST NOT BE TUNED. It is pinned to `NEAR_RADIUS` in views-baseline's
    `matryoshka/geometry.py` — the radius the `near` doll of the eight nested-doll models uses
    (ADR-024, amendment 2026-11-06). This model exists to be that doll's single-level control at
    a moving window, so a different radius would control for something no other model contains.

    ⚠️ Measured by views-baseline on the real 64,818 `land` cells: 8 have no other land cell
    within 5 cells, and for those 8 this model is plain rolling conflictology. See
    `tight_canvas`'s note; the paper should state it rather than let a reader assume otherwise.

    ⛔ S = 64, IN THREE PLACES AT ONCE — here, `n_posterior_samples` below, and
    `config_sweep.py` (C-104, C-85).

    ⚠️ SPEED, measured by views-baseline on a laptop at 64,800 synthetic cells, radius 5, S=64,
    3 targets: about 93-112 s per origin for predict, 2.5 GiB peak — roughly 1.8x
    RollingPooledConflictology. That is a laptop figure on synthetic cells, so it is a rough
    guide and not this model's cost. Measure ONE origin on the rented machine before paying for
    thirteen.

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
        # views-baseline 1.6.0 / #178. REQUIRED and audited; a whole number in 0..359. Must be
        # IDENTICAL to tight_canvas's or the origin-0 equality check cannot pass.
        "radius": 5,
    }

    return hyperparameters
