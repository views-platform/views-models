def get_sweep_config():
    """Return the basic step sweep for silent_orbit."""
    return {
        "method": "grid",
        "name": "silent_orbit",
        "metric": {"name": "MSE", "goal": "minimize"},
        "parameters": {"steps": {"values": [[*range(1, 36 + 1)]]}},
    }
