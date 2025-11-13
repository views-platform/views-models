def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    This configuration is "operational" so modifying it will change the search strategy, parameter ranges, and other settings for hyperparameter tuning aimed at optimizing model performance.

    Returns:
    - sweep_config (dict): A dictionary containing the configuration for hyperparameter sweeps, defining the methods and parameter ranges used to search for optimal hyperparameters.
    """

    sweep_config = {
        "method": "bayes",
        "name": "teenage_dirtbag_tcn_focus",
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 8,  # allow emergence of non-zero predictions
            "eta": 2
        },
        "metric": {
            "name": "time_series_wise_msle_mean_sb",  # MSLE stresses relative error on early growth
            "goal": "minimize"
        },
    }

    parameters_dict = {
        # -------- Temporal Horizon --------
        # Context length balanced: >84 adds memory cost, dilution; keep up to 84 for spatial diffusion patterns.
        "steps": {"values": [[*range(1, 36 + 1)]]},
        "input_chunk_length": {"values": [36, 48, 60, 72, 84]},

        # -------- Training Regime --------
        "batch_size": {"values": [64, 96, 128]},  # Large (>256) averages away rare spike gradients
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [8, 10]},  # Enough patience for late small positives

        # -------- Optimization --------
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 8e-4,  # cap below 1e-3
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 5e-6,
            "max": 2e-4,  # excessive WD suppresses small positives
        },
        "lr_scheduler_factor": {
            "distribution": "uniform",
            "min": 0.35,
            "max": 0.6,
        },
        "lr_scheduler_patience": {"values": [3, 5]},
        "lr_scheduler_min_lr": {"values": [1e-6]},

        # -------- Scaling & Logging --------
        "feature_scaler": {"values": ["RobustScaler"]},
        "target_scaler": {"values": ["RobustScaler"]},
        "log_targets": {"values": [True]},
        "log_features": {
            "values": [["lr_ged_sb", "lr_ged_ns", "lr_ged_os", "lr_acled_sb", "lr_acled_os",
                        "lr_ged_sb_tsum_24", "lr_splag_1_ged_sb", "lr_splag_1_ged_os", "lr_splag_1_ged_ns"]]
        },

        # -------- TCN Architecture --------
        # num_filters == number of temporal convolution blocks (depth); keep moderate to avoid overfitting spikes.
        "num_filters": {"values": [3, 4, 5, 6]},  # >6 rarely improves escalation capture
        # kernel_size: small captures sharp spikes; larger adds smoothing. Exclude extremes >7.
        "kernel_size": {"values": [2, 3, 5, 7]},
        # dilation_base controls expansion of receptive field; avoid >4 (explodes RF, dilutes spikes).
        "dilation_base": {"values": [2, 3]},
        # Dropout moderate: >0.4 erodes rare patterns, <0.1 overfits spikes.
        "dropout": {"values": [0.15, 0.25, 0.35]},
        # Reversible instance norm may stabilize distribution shifts (test both).
        "use_reversible_instance_norm": {"values": [True, False]},

        # -------- Loss / Imbalance Handling --------
        "loss_function": {"values": ["WeightedPenaltyHuberLoss"]},
        # zero_threshold tuned for post log1p + RobustScaler (keeping 1–3 fatalities > zero).
        "zero_threshold": {
            "distribution": "log_uniform_values",
            "min": 3e-4,
            "max": 7e-3,
        },
        # FP penalty capped to avoid suppressing escalation predictions.
        "false_positive_weight": {
            "distribution": "uniform",
            "min": 1.2,
            "max": 3.2,
        },
        # FN penalty emphasizes missed spikes; avoid very high (>6) instability.
        "false_negative_weight": {
            "distribution": "uniform",
            "min": 2.5,
            "max": 5.5,
        },
        # Non-zero weight keeps sparse positive gradients meaningful.
        "non_zero_weight": {
            "distribution": "uniform",
            "min": 4.0,
            "max": 8.0,
        },
        # Huber delta balances sensitivity to moderate spikes vs robustness to extreme outliers.
        "delta": {
            "distribution": "log_uniform_values",
            "min": 0.06,
            "max": 1.6,
        },

        # -------- Gradient Stability --------
        "gradient_clip_val": {
            "distribution": "uniform",
            "min": 0.6,
            "max": 1.4,
        },
    }

    sweep_config["parameters"] = parameters_dict

    return sweep_config