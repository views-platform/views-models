
def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    This configuration is "operational" so modifying it will change the search strategy, parameter ranges, and other settings for hyperparameter tuning aimed at optimizing model performance.

    Returns:
    - sweep_config (dict): A dictionary containing the configuration for hyperparameter sweeps, defining the methods and parameter ranges used to search for optimal hyperparameters.
    """

    sweep_config = {
        'method': 'bayes',
        'name': 'grassy_ditch',
        'run_cap': 30,  # TabICL runs are slow (in-context prediction over the full training set per step)
    }

    # Optimization target: the evaluation stage logs per-target mean metrics as
    # 'month-wise/<target>/<metric>_mean'.
    metric = {
        'name': 'month-wise/lr_ged_sb/CRPS_mean',
        'goal': 'minimize'
    }
    sweep_config['metric'] = metric

    # TabICL settings that change predictions. Compute-only settings (batch_size,
    # kv_cache, device, use_amp, offload_mode, n_jobs) and random_state (pinned
    # from rng_seed) are left out.
    # Both stages are TabICL: clf_* and reg_* keys go to their stage.
    parameters_dict = {
        'hurdle_strategy': {'values': ['probabilistic', 'shares']},
        'clf_n_estimators': {'values': [4, 8, 16]},  # ensemble size; runtime scales linearly
        # None is TabICL's default, an ensemble mixing 'none' and 'power'
        'clf_norm_methods': {'values': [None, 'power', 'quantile', 'robust']},
        'clf_outlier_threshold': {'values': [3.0, 4.0, 6.0, 10.0]},  # z-score clip on features
        # Calibration of P(y > 0), which sets the zero share of the hurdle
        'clf_softmax_temperature': {'min': 0.5, 'max': 1.5, 'distribution': 'uniform'},
        'clf_average_logits': {'values': [True, False]},
        'reg_n_estimators': {'values': [4, 8, 16]},  # ensemble size; runtime scales linearly
        # None is TabICL's default, an ensemble mixing 'none' and 'power'
        'reg_norm_methods': {'values': [None, 'power', 'quantile', 'robust']},
        'reg_outlier_threshold': {'values': [3.0, 4.0, 6.0, 10.0]},  # z-score clip on features
    }
    sweep_config['parameters'] = parameters_dict

    return sweep_config
