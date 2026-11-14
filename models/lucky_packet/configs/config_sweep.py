def get_sweep_config():
    """Return the basic step sweep for lucky_packet."""
    return {
        "method": "grid",
        "name": "lucky_packet",
        "metric": {"name": "MSE", "goal": "minimize"},
        "parameters": {"steps": {"values": [[*range(1, 36 + 1)]]}},
    }
