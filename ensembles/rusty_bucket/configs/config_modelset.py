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
        "models": [
            "violet_visitor",
            "bright_starship",
            "bold_comet",
            "blazing_meteor",
            "heavy_freighter",
            "pink_pirate",
            "blue_stranger",
            "purple_alien",
        ],
    }
    return modelset_config
