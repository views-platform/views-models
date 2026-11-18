
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
                "n_estimators": 300,
                "learning_rate": 0.2,
                "max_depth": 13,
                "num_leaves": 80,
                "reg_alpha": 0.2,
                "reg_lambda": 0.05,
                "n_jobs": -2
            },
            # model_reg: DistributionalRandomForestRegressor (R DRF via rpy2;
            # requires system R + CRAN 'drf') — positive stage.
            "reg": {
                "num_features": 15,
                "num_trees": 500,
                "min_node_size": 12,
                "alpha": 0.05,
                "num_threads": 28
            }
        }
    }
    return hyperparameters
