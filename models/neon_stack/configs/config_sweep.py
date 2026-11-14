def get_sweep_config():
    """Return the basic step sweep for neon_stack."""
    return {
        "method": "grid",
        "name": "neon_stack",
        "metric": {"name": "MSE", "goal": "minimize"},
        "parameters": {"steps": {"values": [[*range(1, 36 + 1)]]}},
    }
