
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
            # model_clf: LGBMClassifier — binary stage, P(lr_ged_sb > 0).
            "clf": {
                "boosting_type": "dart",
                "n_estimators": 200,
                "learning_rate": 0.1,
                "max_depth": 8,
                "num_leaves": 55,
                "reg_alpha": 0.2,
                "reg_lambda": 0.1,
                "n_jobs": -2
            },
            # model_reg: RandomForestQuantileRegressor — positive stage.
            "reg": {
                "n_estimators": 500,
                "max_depth": 12,
                "criterion": "absolute_error",
                "min_samples_split": 10,
                "min_samples_leaf": 65,
                "max_features": "sqrt",
                "n_jobs": -2
            }
        }
    }
    return hyperparameters
