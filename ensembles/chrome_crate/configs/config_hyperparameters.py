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

    ⛔ READ THIS BEFORE SIZING A BOX — AND NOTE WHAT THIS FILE USED TO SAY.

    **chrome_crate is country-month.** Its pool is tiny, and this block existed for ten
    days claiming the opposite, because it was copied from `chrome_box` — the pgm twin —
    and the grid arithmetic was never replaced. Until 2026-10-30 it reasoned about
    "64,742 cells x 36 steps = 2,330,712 rows", announced
    "**PEAK IS ~53.8 GiB ... `RAM >= 50 GB` IS NOT ENOUGH**", and concluded
    "87 GiB -- this is the class to rent". For a seven-minute job.

    That is wrong by a factor of roughly 570, and it is wrong in the expensive direction:
    anyone sizing phase 1 from it over-rents by two machine classes and reads "DOES NOT
    FIT" about a workload that never strained a box. Found by an independent /falsify,
    2026-10-30. The failure class is the one this epic keeps producing — a correct value
    answering the wrong question — and here it arrived by copy-paste between twins.

    **The arithmetic for THIS ensemble, cm:**

        191 countries x 36 steps                  = 6,876 rows
        x 1,024 draws x 4 bytes (float32)         = 26.9 MiB per target
        x 3 regression targets                    = 80.6 MiB pooled
        model_results, held beside it             = 80.6 MiB
        ---------------------------------------------------------
        frame arithmetic                          ~0.16 GiB

    **Measured on a pod, 2026-10-28** — `tools/podrun/pod_run_crafd_chain.sh:186-190`,
    from the run's own MANIFEST:

        chrome_crate pool, on disk                      96 MB
        the eight cm members, retained                 176 MB
        each member's peak RSS                        ~2.0 MB
        whole phase, wall clock                          7 min
        box it ran on                           87 GiB visible, floor 16, NEVER STRAINED

    So the arithmetic and the measurement agree, which is the point: 0.16 GiB computed
    against 96 MB + 176 MB observed. `MEM_FLOOR_GB=16` in the runner is not a tight floor
    and does not need to be — it exists to refuse a box too small for the venv's working
    set and the datafactory cache, not for the pool.

    ⛔ **The 53.8 GiB / 87 GiB figures are chrome_box's and belong only there.**
    `chrome_box` is pgm global-land: 64,742 cells rather than 191 countries, measured at
    **>= 68.2 GiB** on the same day, and it genuinely wants the 87 GiB class. Those numbers
    live in `ensembles/chrome_box/configs/config_hyperparameters.py` with their provenance
    and their lower-bound caveat, pinned by
    `tests/test_chrome_box_memory_is_the_measured_peak.py`. Do not copy them back here;
    that is how this defect happened.

    ⚠️ The draw count is NOT a memory problem for this ensemble. 8 x 128 = 1,024 draws is
    what epic #595 specifies and what the reconciliation needs on both sides to reach
    ALIGNED_DRAWS rather than POINT_BROADCAST, and at cm resolution it costs 81 MiB. The
    `rusty_bucket` precedent — "Thinned from 8x128=1024 on 2026-07-20: the full-S run peaks
    ~28.6 GB and cannot fit production hardware" — is a **pgm** measurement about the laptop
    rotation (C-99), and it is recorded here only so the next reader does not reapply it to
    a country-month pool. It constrains chrome_box's phase, not this one.

    Returns:
    - hp_config (dict): A dictionary containing ensemble hyperparameters.
    """
    hp_config = {
        "steps": [*range(1, 36 + 1, 1)],
        "expected_models": 8,
        "expected_samples_per_model": 128,
    }
    return hp_config
