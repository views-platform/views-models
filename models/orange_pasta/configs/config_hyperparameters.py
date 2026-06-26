def get_hp_config(): 
    hp_config = {
        "target_transform": "identity",
        "steps": [*range(1, 36 + 1, 1)],
        "time_steps": 36,
        "parameters": {
            "n_estimators": 200,
            "n_jobs": 12,
        }
    }
    return hp_config