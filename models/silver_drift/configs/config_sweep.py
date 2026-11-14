def get_sweep_config():
    """Return the basic step sweep for silver_drift."""
    return {
        "method": "grid",
        "name": "silver_drift",
        "metric": {"name": "MSE", "goal": "minimize"},
        "parameters": {"steps": {"values": [[*range(1, 36 + 1)]]}},
    }
