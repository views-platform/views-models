
def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    This configuration is "operational" so modifying it will change the search strategy, parameter ranges, and other settings for hyperparameter tuning aimed at optimizing model performance.

    Returns:
    - sweep_config (dict): A dictionary containing the configuration for hyperparameter sweeps, defining the methods and parameter ranges used to search for optimal hyperparameters.
    """

    sweep_config = {
        'method': 'bayes',
        'name': 'rocky_rocks',
        'run_cap': 30
    }

    # Optimization target: the evaluation stage logs per-target mean metrics as
    # 'month-wise/<target>/<metric>_mean'. CRPS matches the model's purpose
    # (distributional positive stage, score='CRPScore'); twCRPS_mean is a
    # threshold-weighted alternative logged by the same stage.
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
        # clf: XGBClassifier
        'clf_n_estimators': {'min': 400, 'max': 800, 'distribution': 'int_uniform'},
        'clf_learning_rate': {'values': [0.01, 0.05, 0.1]},
        'clf_max_depth': {'min': 5, 'max': 13, 'distribution': 'int_uniform'},
        'clf_gamma': {'min': 7, 'max': 11},
        'clf_subsample': {'values': [0.6, 0.7, 0.8, 0.9, 1.0]},
        'clf_colsample_bytree': {'values': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]},
        'clf_reg_alpha': {'min': 10, 'max': 40, 'distribution': 'int_uniform'},
        'clf_reg_lambda': {'min': 0.5, 'max': 2.5},
        'clf_min_child_weight': {'values': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]},
        'clf_max_delta_step': {'min': 3, 'max': 8},
        'clf_n_jobs': {'values': [28]},
        # reg: NGBoostRegressor (NGBRegressorWrapper)
        'reg_dist': {'values': ['LogNormal']},
        'reg_score': {'values': ['CRPScore']},
        'reg_n_estimators': {'min': 300, 'max': 600, 'distribution': 'int_uniform'},
        'reg_learning_rate': {'values': [0.005, 0.01, 0.02, 0.05]},
        'reg_base_criterion': {'values': ['squared_error']},
        'reg_base_splitter': {'values': ['random']},
        'reg_base_max_depth': {'min': 2, 'max': 4, 'distribution': 'int_uniform'},
        'reg_minibatch_frac': {'values': [0.6, 0.7, 0.8, 0.9, 1.0]},
        'reg_col_sample': {'values': [0.3, 0.4, 0.5, 0.6, 0.7]},
    }
    sweep_config['parameters'] = parameters_dict

    return sweep_config
