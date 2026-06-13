def get_hp_config(): 
    hp_config = {
        "target_transform": "identity",
        "steps": [*range(1, 36 + 1, 1)],
        "time_steps": 36,
        "parameters": {
            "clf":{
                "n_estimators": 200,
            },
            "reg":{
                "n_estimators": 200,
            }
        }
    }
    return hp_config