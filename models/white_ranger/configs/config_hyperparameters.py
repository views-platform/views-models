def get_hp_config():
    """
    Contains the hyperparameter configurations for the baseline model.
    This configuration is "operational" so modifying these settings will impact the model's behavior.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for the baseline model.
    """

    hyperparameters = {
        "regression_targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
        "steps": list(range(1, 37)),
        "time_steps": 36,
        "window_months": 36,
        "n_samples": 64,
        "seed": 42,
    }

    return hyperparameters
