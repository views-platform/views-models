def get_hp_config():
    """Hyperparameters for chrome_box — epic #595's grid cells (64,742 at land_gaul) pool.

    `expected_models` and `expected_samples_per_model` are the ADR-015 contract, and
    `tests/test_ensemble_configs.py` asserts them against reality: `expected_models` must
    equal `len(config_modelset["models"])`, and every constituent's produced count
    (`n_posterior_samples` x `n_head_samples`) must equal `expected_samples_per_model`.
    So these are guards, not documentation — a thinned member or a dropped one fails CI.

    PFE concat concatenates draws on the sample axis, so the pooled total is
    `expected_models` x `expected_samples_per_model` = 8 x 128 = **1,024 draws**, which is
    what epic #595 specifies and what the reconciliation needs on both sides to reach
    ALIGNED_DRAWS rather than POINT_BROADCAST.

    ⛔ READ THIS BEFORE LAUNCHING A RUN. 8 x 128 is the configuration `rusty_bucket`
    already tried and abandoned. Its own config records:

        "Thinned from 8x128=1024 on 2026-07-20: the full-S run peaks ~28.6 GB and cannot
         fit production hardware -- see views-baseline memory issue + C-99."

    The arithmetic reproduces that figure exactly, for this ensemble's shape:

        64,742 cells x 36 steps                   = 2,330,712 rows
        x 1,024 draws x 4 bytes (float32)         = 9.55 GB per target
        x 3 regression targets                    = 28.6 GB
        rusty_bucket as thinned (8 x 16 = 128)    = 1.19 GB per target, 3.6 GB for three

    `chunk_by_time=True` (the default in `reconcile_frames`, register C-200a) keeps the
    RECONCILIATION from holding a global-volume x S frame whole. It does not shrink the
    concat pool itself, so it does not answer this.

    The number here is the epic's, deliberately, so the decision is visible rather than
    pre-made: run it on hardware that can hold it, or thin to 8 x 16 and amend #595's
    acceptance. That is a hardware and cost call, not a config one.

    Returns:
    - hp_config (dict): A dictionary containing ensemble hyperparameters.
    """
    hp_config = {
        "steps": [*range(1, 36 + 1, 1)],
        "expected_models": 8,
        "expected_samples_per_model": 128,
    }
    return hp_config
