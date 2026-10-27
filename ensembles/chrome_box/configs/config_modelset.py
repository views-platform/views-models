def get_modelset_config():
    """The eight pgm distributional baselines of epic #595, in the epic's table order.

    ⛔ THE ORDER IS LOAD-BEARING, not cosmetic. `aggregation: "concat"` lays member *k*'s
    draws at sample-axis positions 128(k-1)..128k-1, and the reconciliation runs in
    ALIGNED_DRAWS mode, which scales grid draw *i* to country draw *i*. So position *k* in
    this list must hold the same (algorithm, window_months, lambda_mix) setting as position
    *k* in chrome_crate's list, or every draw is reconciled against an
    unrelated model's country total — silently, with a plausible number coming out.

    tests/test_chrome_box_pairing.py pins this list element by element against its twin.
    Reordering either list fails there; it does not fail at run time.

    Returns:
    - modelset_config (dict): A dictionary with the key 'models' listing constituent model names.
    """
    modelset_config = {
        "models": [
            # algorithm            window_months  lambda_mix
            "ashen_sparrow",   # ConflictologyModel      12          -
            "brisk_sparrow",   # ConflictologyModel      18          -
            "candid_sparrow",  # ConflictologyModel      24          -
            "dusky_sparrow",   # ConflictologyModel      36          -
            "eager_sparrow",   # MixtureBaseline         18         0.1
            "faint_sparrow",   # MixtureBaseline         36         0.1
            "gentle_sparrow",  # MixtureBaseline         18         0.3
            "humble_sparrow",  # MixtureBaseline         36         0.3
        ],
    }
    return modelset_config
