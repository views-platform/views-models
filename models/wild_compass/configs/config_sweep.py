def get_sweep_config():
    """Return the basic step sweep for wild_compass."""
    return {
        "method": "grid",
        "name": "wild_compass",
        "metric": {"name": "MSE", "goal": "minimize"},
        "parameters": {"steps": {"values": [[*range(1, 36 + 1)]]}},
    }
