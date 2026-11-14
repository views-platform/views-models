def get_sweep_config():
    """Return the basic step sweep for hidden_peak."""
    return {
        "method": "grid",
        "name": "hidden_peak",
        "metric": {"name": "MSE", "goal": "minimize"},
        "parameters": {"steps": {"values": [[*range(1, 36 + 1)]]}},
    }
