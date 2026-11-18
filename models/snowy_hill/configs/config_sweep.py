
def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    This configuration is "operational" so modifying it will change the search strategy, parameter ranges, and other settings for hyperparameter tuning aimed at optimizing model performance.

    Returns:
    - sweep_config (dict): A dictionary containing the configuration for hyperparameter sweeps, defining the methods and parameter ranges used to search for optimal hyperparameters.
    """

    sweep_config = {
        'method': 'bayes',
        'name': 'snowy_hill',
        'run_cap': 30
    }

    # Optimization target: the evaluation stage logs per-target mean metrics as
    # 'time-series-wise/<target>/<metric>_mean'. CRPS matches the model's purpose
    # (distributional positive stage); twCRPS_mean is a threshold-weighted
    # alternative logged by the same stage.
    metric = {
        'name': 'time-series-wise/lr_ged_sb/CRPS_mean',
        'goal': 'minimize'
    }
    sweep_config['metric'] = metric

    # In sweep mode the estimators receive ONLY the flat clf_*/reg_* keys below
    # (config_hyperparameters "parameters" is ignored, and the manager splits
    # the flat keys into clf/reg dicts), so every estimator argument must be
    # declared here. Fixed single-value entries pin the remaining choices.
    parameters_dict = {
        'hurdle_strategy': {'values': ['probabilistic', 'shares']},
        # clf: RandomForestClassifier
        'clf_n_estimators': {'min': 300, 'max': 700, 'distribution': 'int_uniform'},
        'clf_max_depth': {'min': 10, 'max': 20, 'distribution': 'int_uniform'},
        'clf_min_samples_split': {'min': 2, 'max': 7, 'distribution': 'int_uniform'},
        'clf_min_samples_leaf': {'min': 10, 'max': 20, 'distribution': 'int_uniform'},
        'clf_n_jobs': {'values': [28]},
        # reg: RandomForestQuantileRegressor
        'reg_n_estimators': {'min': 500, 'max': 1500, 'distribution': 'int_uniform'},
        'reg_max_depth': {'min': 6, 'max': 13, 'distribution': 'int_uniform'},
        'reg_criterion': {'values': ['absolute_error']},
        'reg_min_samples_split': {'min': 8, 'max': 25, 'distribution': 'int_uniform'},
        'reg_min_samples_leaf': {'min': 40, 'max': 70, 'distribution': 'int_uniform'},
        'reg_max_features': {'values': ['sqrt']},
        'reg_n_jobs': {'values': [28]},
    }
    sweep_config['parameters'] = parameters_dict

    return sweep_config
