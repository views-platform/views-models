def get_modelset_config():
    """
    Contains the list of constituent models for the ensemble.

    Returns:
    - modelset_config (dict): A dictionary with the key 'models' listing constituent model names.

    The Epic #242 roster, LOCKED in the 05 pre-registration (views-hydranet#246) and
    pinned in `tests/test_roster_conformance.py`:

        gated_NB     (nb,         soft_gate)           violet_visitor / bright_starship / bold_comet
        th_gated_NB  (nb,         threshold_gate 0.5)  blazing_meteor / heavy_freighter
        mixture_NB   (mixture_nb, soft_gate)           pink_pirate / blue_stranger / purple_alien

    That table is the pre-registration as locked, kept as the record. It is SUPERSEDED by
    the 2026-09 reconfiguration (#463) and the gate-threshold priors (#466); the live
    per-member values are ``ROSTER`` in `tests/test_roster_conformance.py`.

    These replace the eight `temporary_*` stand-ins — clones of the `heavy_strider`
    global-land baseline, a degenerate mixture that existed to exercise the pooled-draw
    machinery at the right shape while the real models were built (#146). They have done
    that job.

    Every member emits D x K = 4 x 4 = 16 draws, so the pool is 8 x 16 = 128 and each
    constituent carries equal weight (ADR-015 §2/§3, §6). That uniformity is why this swap
    could not happen until violet_visitor's sample count was settled: it emitted 8, and
    the config-time contract correctly refused the mismatch rather than pooling unequally.
    """
    modelset_config = {
        # Order revised 2026-09-07 with the roster (views-hydranet #324). Membership is unchanged
        # -- the same eight models -- but the order now matches ROSTER in
        # tests/test_roster_conformance.py, which compares the two as ordered lists. Concat pooling
        # is order-independent, so this changes no forecast; it keeps the two declarations of the
        # roster from drifting apart, which is the whole point of that test.
        "models": [
            "purple_alien",
            "pink_pirate",
            "blue_stranger",
            "bold_comet",
            "blazing_meteor",
            "heavy_freighter",
            "bright_starship",
            "violet_visitor",
        ],
    }
    return modelset_config
