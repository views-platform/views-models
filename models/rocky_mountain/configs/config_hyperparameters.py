
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
            # model_clf: XGBClassifier — binary stage, P(lr_ged_sb > 0).
            "clf": {
                "n_estimators": 500,
                "learning_rate": 0.1,
                "max_depth": 9,
                "gamma": 7,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 20,
                "reg_lambda": 2.4,
                "min_child_weight": 0.5,
                "max_delta_step": 8,
                "n_jobs": -2
            },
            # model_reg: RandomForestQuantileRegressor — positive stage.
            "reg": {
                "n_estimators": 500,
                "max_depth": 12,
                "criterion": "absolute_error",
                "min_samples_split": 18,
                "min_samples_leaf": 69,
                "max_features": "sqrt",
                "n_jobs": -2
            }
        }
    }
    return hyperparameters
