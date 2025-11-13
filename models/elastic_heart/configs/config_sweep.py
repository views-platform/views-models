def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    This configuration is "operational" so modifying it will change the search strategy, parameter ranges, and other settings for hyperparameter tuning aimed at optimizing model performance.

    Returns:
    - sweep_config (dict): A dictionary containing the configuration for hyperparameter sweeps, defining the methods and parameter ranges used to search for optimal hyperparameters.
    """

    sweep_config = {
        "method": "bayes",
        "name": "elastic_heart_tsmixer_focus",
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 10,     # allow emergence of non-zero predictions
            "eta": 2
        },
        "metric": {
            "name": "time_series_wise_msle_mean_sb",
            "goal": "minimize"
        },
    }

    parameters_dict = {
        # ---------------- Temporal Context ----------------
        # Limit max context (>=84 often dilutes sharp pre-spike patterns & raises memory).
        "steps": {"values": [[*range(1, 36 + 1)]]},
        "input_chunk_length": {"values": [36, 48, 60, 72]},  # 84 removed (diminishing returns)

        # ---------------- Training Regime ----------------
        "batch_size": {"values": [64, 96, 128]},  # Mid sizes keep rare spike windows in batches
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [10]},  # Longer patience to catch late signal emergence

        # ---------------- Optimization ----------------
        # LR capped below 1e-3 to avoid rapid convergence to trivial zero predictions.
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 6e-4,
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 5e-6,
            "max": 2e-4,  # Heavy regularization suppresses small positives
        },
        "lr_scheduler_factor": {
            "distribution": "uniform",
            "min": 0.35,
            "max": 0.6,
        },
        "lr_scheduler_patience": {"values": [3, 5]},
        "lr_scheduler_min_lr": {"values": [1e-6]},

        # ---------------- Scaling & Logging ----------------
        # RobustScaler preferred: median/IQR preserves low positive separation after log1p.
        "feature_scaler": {
            "values": ["RobustScaler"]  # Single scaler for consistency (ablation removed)
        },
        "target_scaler": {
            "values": ["RobustScaler"]  # Single target scaler per requirement
        },
        "log_targets": {"values": [True]},  # log1p expands 1–5 range; retains spike ratios
        "log_features": {
            "values": [["lr_ged_sb", "lr_ged_ns", "lr_ged_os", "lr_acled_sb", "lr_acled_os", "lr_ged_sb_tsum_24", 
                         "lr_splag_1_ged_sb", "lr_splag_1_ged_os", "lr_splag_1_ged_ns"]]
        },

        # ---------------- Architecture (Spike Sensitivity) ----------------
        # Depth: too many blocks overfit rare extreme spikes → restrict.
        "num_blocks": {"values": [2, 3, 4]},  # Removed 5,6
        # ff_size & hidden_size upper bound trimmed (512 often memorizes spikes only).
        "ff_size": {"values": [64, 128, 256]},
        "hidden_size": {"values": [64, 128, 192, 256]},
        # Activations: LeakyReLU aids gradient flow near zero; GELU smoother; ELU mitigates vanishing.
        "activation": {
            "values": ["LeakyReLU", "GELU", "ReLU", "ELU"]
        },
        # Dropout moderate: >0.4 erases rare patterns; <0.1 overfits spikes.
        "dropout": {"values": [0.15, 0.25, 0.35]},
        # Normalization variants retained; removing TimeBatchNorm2d would reduce temporal adaptivity.
        "norm_type": {
            "values": ["LayerNorm", "LayerNormNoBias", "TimeBatchNorm2d"]
        },
        "normalize_before": {"values": [True, False]},

        # ---------------- Loss & Imbalance Handling ----------------
        "loss_function": {"values": ["WeightedPenaltyHuberLoss"]},
        # Zero threshold range tuned post log+robust scaling so 1–3 fatalities remain positive.
        "zero_threshold": {
            "distribution": "log_uniform_values",
            "min": 3e-4,
            "max": 7e-3,
        },
        # FP penalty capped to avoid suppressing emergence of new conflict.
        "false_positive_weight": {
            "distribution": "uniform",
            "min": 1.2,
            "max": 3.2,
        },
        # FN penalty encourages detecting spikes; upper bound restrained to prevent instability.
        "false_negative_weight": {
            "distribution": "uniform",
            "min": 2.5,
            "max": 5.5,
        },
        # Non-zero emphasis stabilizes gradient signal from sparse positives.
        "non_zero_weight": {
            "distribution": "uniform",
            "min": 4.0,
            "max": 8.0,
        },
        # Huber delta controls sensitivity to medium spikes (avoid huge linear region).
        "delta": {
            "distribution": "log_uniform_values",
            "min": 0.06,
            "max": 1.5,
        },

        # ---------------- Gradient Stability ----------------
        "gradient_clip_val": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1.4,
        },
    }

    sweep_config['parameters'] = parameters_dict

    return sweep_config