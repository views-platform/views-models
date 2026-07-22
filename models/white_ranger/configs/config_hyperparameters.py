def get_hp_config():
    """
    Contains the hyperparameter configurations for the baseline model.
    This configuration is "operational" so modifying these settings will impact the model's behavior.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for the baseline model.
    """

    hyperparameters = {
        "regression_targets": ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"],
        "steps": list(range(1, 37)),
        "time_steps": 36,
        "window_months": 36,
        "n_samples": 64,
        "n_posterior_samples": 64,
        "seed": 42,
        "skip_predictions_delivery": True,
    }

    return hyperparameters
