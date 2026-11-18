
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
                "n_estimators": 250,
                "learning_rate": 0.2,
                "max_depth": 10,
                "num_leaves": 50,
                "reg_alpha": 0.1,
                "reg_lambda": 0.05,
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
