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

    ⚠️ "production hardware" THERE MEANS THE LAPTOP ROTATION (C-99), NOT A POD -- so the
    epic's draw count is not the blocker it first reads as. But the pool is NOT the peak,
    and the peak is what decides the pod.

    ⛔ **PEAK IS 68.2 GiB, MEASURED. `RAM >= 50 GB` IS NOT ENOUGH.**
    Measured on a pod 2026-10-28 by sampling cgroup `anon` at 30s through pool and
    reconcile, on a run that completed. The frame arithmetic below predicted 53.86 GiB
    and was right about the frames; the remaining 14.3 GiB (1.27x) is the transient
    allocation this comment already flagged as unmeasured, at the bottom. Two figures
    reported during that session were wrong and are recorded here so neither is
    remembered as live: **19.8 GiB** was `memory.current`, which includes reclaimable
    page cache; **27.8 GiB** was a 60s-interval undersample, and sits below the frame
    floor, which is how it was caught — a number below the floor cannot be a peak.
    Traced through `prediction_frame_ensemble.py::_forecast_ensemble` (pipeline-core
    3.3.0), which builds `model_results` for ALL constituents and ALL targets before the
    aggregation loop, frees it nowhere -- no `del`, no `gc.collect` -- and then accumulates
    the pooled frames beside it in `forecasts[target]`:

        model_results  8 members x 3 targets x 128 draws        26.67 GiB
        forecasts      3 targets x 1,024 pooled draws           26.67 GiB
        input volume   64,742 x 441 months x 4 features, f32     0.43 GiB
        cm frame       ~200 countries x 36 x 1,024 x 3           0.08 GiB
        ------------------------------------------------------ ----------
        FRAME FLOOR (not the peak -- see above)                 53.86 GiB
        measured peak, 2026-10-28                               68.20 GiB

    Against the written selection rule and the machines the 2026-09 campaign actually used:

        50 GiB (the rule)   DOES NOT FIT
        57 GiB (used)       120% -- DOES NOT FIT EITHER
        87 GiB (used)       78% -- this is the class to rent, and it is not roomy

    ⛔ The 62% this table claimed for the 87 GiB box until 2026-10-28 was the FRAME FLOOR
    divided by the box, not the peak. Anyone sizing from it would have under-rented. The
    rule the mistake breaks: a floor is not a peak, and only the measurement settles the
    gap.

    ⚠️ 94% is not "fits". The 31 GB machine the runpod postmortem says "do not take"
    collapsed 25x -- 32 steps/s to 0.11 -- by APPROACHING its ceiling at ~20 of 31 GB, from
    kernel reclaim, never an OOM. Nothing failed; it only slowed, for eight hours.
    **Headroom is the property, not fit.**

    ⚠️ And `docs/runpod_run_guide.md`: "the in-code memory guard cannot see the container.
    It reads the host's RAM and will approve a run that cannot fit." So this arithmetic is
    done by hand whatever the pod reports.

    ⚠️ Unit note, because the two conventions differ by 7% right at the boundary:
    `rusty_bucket`'s "~28.6 GB" is DECIMAL and this file's 26.67 is GiB -- the same
    quantity. The run guide's own check is `memory.max / 1073741824`, i.e. GiB, so the
    figures above are GiB and comparable to it.

    ⛔ Transient allocations inside `np.concatenate` and inside the reconciler beyond the
    frames counted above are now bounded by subtraction, not by analysis: 68.2 measured
    minus 53.86 of frames = ~14.3 GiB. Still not attributed to a line of code.

    ⛔ The chunking that bounds reconciliation is NOT `chunk_by_time`. The measured run
    logged `reconcile_frames: mode=aligned-draws (2330712 grid rows, 1024 draws) --
    per-draw approximation (C-200b)`, three times, once per target. So the reconciler
    held one draw's grid at a time, not one month's -- which is why the sizing argument
    must rest on the 68.20 measurement and not on reasoning about the time axis. The
    53.86 figure remains a floor on the peak and the 87 GiB class remains the one to
    rent, but the reason is now a measurement rather than an inference.

    If the 87 GiB class is unavailable, the honest alternatives are to thin to 8 x 16 as
    `rusty_bucket` did and amend #595's acceptance, or to reduce the target set -- either
    of which is a decision about what the epic delivers, not a config tidy-up.

    Returns:
    - hp_config (dict): A dictionary containing ensemble hyperparameters.
    """
    hp_config = {
        "steps": [*range(1, 36 + 1, 1)],
        "expected_models": 8,
        "expected_samples_per_model": 128,
    }
    return hp_config
