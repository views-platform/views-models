def get_sweep_config():
    """Return the basic step sweep for golden_rune."""
    return {
        "method": "grid",
        "name": "golden_rune",
        "metric": {"name": "MSE", "goal": "minimize"},
        "parameters": {"steps": {"values": [[*range(1, 36 + 1)]]}},
    }
