
def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    This configuration is "operational" so modifying it will change the search strategy, parameter ranges, and other settings for hyperparameter tuning aimed at optimizing model performance.

    Returns:
    - sweep_config (dict): A dictionary containing the configuration for hyperparameter sweeps, defining the methods and parameter ranges used to search for optimal hyperparameters.
    """

    sweep_config = {
        'method': 'bayes',
        'name': 'snowy_cliff',
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
        'clf_n_estimators': {'min': 300, 'max': 500, 'distribution': 'int_uniform'},
        'clf_max_depth': {'min': 10, 'max': 20, 'distribution': 'int_uniform'},
        'clf_min_samples_split': {'min': 2, 'max': 7, 'distribution': 'int_uniform'},
        'clf_min_samples_leaf': {'min': 10, 'max': 20, 'distribution': 'int_uniform'},
        'clf_n_jobs': {'values': [28]},
        # reg: NGBoostRegressor (NGBRegressorWrapper)
        'reg_dist': {'values': ['LogNormal']},
        'reg_score': {'values': ['CRPScore']},
        'reg_n_estimators': {'min': 300, 'max': 500, 'distribution': 'int_uniform'},
        'reg_learning_rate': {'values': [0.005, 0.01, 0.02, 0.05]},
        'reg_base_criterion': {'values': ['squared_error']},
        'reg_base_splitter': {'values': ['random']},
        'reg_base_max_depth': {'min': 2, 'max': 4, 'distribution': 'int_uniform'},
        'reg_minibatch_frac': {'values': [0.6, 0.7, 0.8, 0.9, 1.0]},
        'reg_col_sample': {'values': [0.3, 0.4, 0.5, 0.6, 0.7]},
    }
    sweep_config['parameters'] = parameters_dict

    return sweep_config
