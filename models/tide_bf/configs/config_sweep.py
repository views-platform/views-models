def get_sweep_config():
    """Return the basic step sweep for tide_bf."""
    return {
        "method": "grid",
        "name": "tide_bf",
        "metric": {"name": "MSE", "goal": "minimize"},
        "parameters": {"steps": {"values": [[*range(1, 36 + 1)]]}},
    }
