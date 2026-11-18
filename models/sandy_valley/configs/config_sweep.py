
def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    This configuration is "operational" so modifying it will change the search strategy, parameter ranges, and other settings for hyperparameter tuning aimed at optimizing model performance.

    Returns:
    - sweep_config (dict): A dictionary containing the configuration for hyperparameter sweeps, defining the methods and parameter ranges used to search for optimal hyperparameters.
    """

    sweep_config = {
        'method': 'bayes',
        'name': 'sandy_valley',
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
        # clf: LGBMClassifier
        'clf_boosting_type': {'values': ['gbdt', 'dart']},
        'clf_max_depth': {'min': 5, 'max': 13, 'distribution': 'int_uniform'},
        'clf_n_estimators': {'min': 100, 'max': 400, 'distribution': 'int_uniform'},
        'clf_learning_rate': {'values': [0.01, 0.05, 0.1, 0.2]},
        'clf_num_leaves': {'min': 40, 'max': 80, 'distribution': 'int_uniform'},
        'clf_reg_alpha': {'values': [0.05, 0.1, 0.2, 0.3]},
        'clf_reg_lambda': {'values': [0.01, 0.05, 0.1,]},
        'clf_n_jobs': {'values': [28]},
        # reg: TabICLQuantileRegressor
        # None is TabICL's default, an ensemble mixing 'none' and 'power'
        'reg_n_estimators': {'values': [4, 8, 16]},  # ensemble size; runtime scales linearly
        'reg_norm_methods': {'values': [None, 'power', 'quantile', 'robust']},
        'reg_outlier_threshold': {'values': [3.0, 4.0, 6.0, 10.0]},  # z-score clip on features
    }
    sweep_config['parameters'] = parameters_dict

    return sweep_config
