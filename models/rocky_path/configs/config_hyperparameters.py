
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
        "hurdle_strategy": "probabilistic",
        "target_transform": "identity",
        "parameters": {
            # model_clf: XGBClassifier — binary stage, P(lr_ged_sb > 0).
            "clf": {
                "n_estimators": 770,
                "learning_rate": 0.1,
                "max_depth": 8,
                "gamma": 9,
                "subsample": 0.7,
                "colsample_bytree": 0.9,
                "reg_alpha": 35,
                "reg_lambda": 2,
                "min_child_weight": 0.8,
                "max_delta_step": 6,
                "n_jobs": -2
            },
            # model_reg: DistributionalRandomForestRegressor (R DRF via rpy2;
            # requires system R + CRAN 'drf') — positive stage.
            "reg": {
                "num_features": 15,
                "num_trees": 550,
                "min_node_size": 5,
                "alpha": 0.1,
                "num_threads": 15
            }
        }
    }
    return hyperparameters
