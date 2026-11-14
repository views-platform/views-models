def get_sweep_config():
    """Return the basic step sweep for fuzzy_logic."""
    return {
        "method": "grid",
        "name": "fuzzy_logic",
        "metric": {"name": "MSE", "goal": "minimize"},
        "parameters": {"steps": {"values": [[*range(1, 36 + 1)]]}},
    }
