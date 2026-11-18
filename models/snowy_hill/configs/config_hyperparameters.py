
def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.
    This configuration is "operational" so modifying these settings will impact the model's behavior during the training.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for training the model, which determine the model's behavior during the training phase.
    """
    
    hyperparameters = {
        "rng_seed": 42,
        "steps": [*range(1, 36 + 1, 1)],
        "time_steps": 36,
        "pred_samples": 500,
        "draw_order": "sorted",
        "skip_predictions_delivery": True,
        "hurdle_strategy": "shares",
        "target_transform": "identity",
        "parameters": {
            # model_clf: RandomForestClassifier — binary stage, P(lr_ged_sb > 0).
            "clf": {
                "n_estimators": 400,
                "max_depth": 14,
                "min_samples_split": 5,
                "min_samples_leaf": 17,
                "n_jobs": -2
            },
            # model_reg: RandomForestQuantileRegressor — positive stage.
            "reg": {
                "n_estimators": 600,
                "max_depth": 13,
                "criterion": "absolute_error",
                "min_samples_split": 23,
                "min_samples_leaf": 70,
                "max_features": "sqrt",
                "n_jobs": -2
            }
        }
    }
    return hyperparameters
