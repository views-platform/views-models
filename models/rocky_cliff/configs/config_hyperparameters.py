
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
                "max_depth": 8,
                "gamma": 9,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "reg_alpha": 35,
                "reg_lambda": 1.0,
                "min_child_weight": 0.7,
                "max_delta_step": 5,
                "n_jobs": -2
            },
            # model_reg: NGBoostRegressor (NGBRegressorWrapper) — positive stage,
            # fit on positive rows only. No n_jobs: NGBoost is single-threaded.
            "reg": {
                "dist": "LogNormal",
                "score": "CRPScore",
                "n_estimators": 500,
                "learning_rate": 0.01,
                "base_criterion": "squared_error",
                "base_splitter": "random",
                "base_max_depth": 2,
                "minibatch_frac": 0.9,
                "col_sample": 0.5
            }
        }
    }
    return hyperparameters
