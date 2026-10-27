def get_hp_config():
    """Hyperparameters for chrome_crate — epic #595's countries pool.

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

    ⚠️ BUT "production hardware" THERE MEANS THE LAPTOP ROTATION (C-99), NOT A POD. The
    rented pods this platform uses are specified in `docs/runpod_run_guide.md` at
    **RAM >= 50 GB, vCPU >= 12, VRAM >= 24 GB**, and the 2026-09 campaign ran on 87 GB and
    57 GB machines. 28.6 GB fits those with room, and the register entry recording this
    drift says the exit is "a run that fits the hardware, not a config edit". So the
    epic's number stands and needs no decision: size the pod on the written rule.

    Two things that DO carry to a run:

    - The 31 GB machine the runpod postmortem says "do not take" failed by APPROACHING its
      ceiling -- a 25x slowdown from kernel reclaim, never an OOM. Headroom is the
      property, not fit.
    - `docs/runpod_run_guide.md`: "the in-code memory guard cannot see the container. It
      reads the host's RAM and will approve a run that cannot fit." So do the arithmetic by
      hand whatever the pod reports.

    The 28.6 GB above is the concat pool ONLY. Peak also carries the input volume and the
    cm frame, and that total is NOT MEASURED -- comfortable on 87 GB, worth measuring
    before taking a 50 GB machine.

    Returns:
    - hp_config (dict): A dictionary containing ensemble hyperparameters.
    """
    hp_config = {
        "steps": [*range(1, 36 + 1, 1)],
        "expected_models": 8,
        "expected_samples_per_model": 128,
    }
    return hp_config
