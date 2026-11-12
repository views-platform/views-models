def get_hp_config():
    hyperparameters = {
        "steps": [*range(1, 36 + 1, 1)],
        "time_steps": 36,
        "skip_predictions_delivery": True,
        "regression_targets": ["synth_target"],
    }
    return hyperparameters
