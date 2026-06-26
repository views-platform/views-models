def get_modelset_config():
    """
    Contains the list of constituent models for the ensemble.

    Returns:
    - modelset_config (dict): A dictionary with the key 'models' listing constituent model names.

    Note: these eight ``temporary_*`` constituents are interim stand-ins (clones of
    the ``heavy_strider`` global-land datafactory baseline). They make a degenerate
    mixture by design — the job now is to exercise the pooled-draw machinery at the
    correct global-land shape. They retire when the real ~8 global HydraNets land (#146).
    """
    modelset_config = {
        "models": [
            "temporary_otter",
            "temporary_robin",
            "temporary_finch",
            "temporary_heron",
            "temporary_lynx",
            "temporary_bison",
            "temporary_crane",
            "temporary_fox",
        ],
    }
    return modelset_config
