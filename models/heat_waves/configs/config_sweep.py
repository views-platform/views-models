def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    This configuration is "operational" so modifying it will change the search strategy, parameter ranges, and other settings for hyperparameter tuning aimed at optimizing model performance.

    Returns:
    - sweep_config (dict): A dictionary containing the configuration for hyperparameter sweeps, defining the methods and parameter ranges used to search for optimal hyperparameters.
    """

    sweep_config = {
        "method": "bayes",
        "name": "heat_waves_tft_focus",
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 10,  # allow emergence of non-zero predictions
            "eta": 2
        },
        "metric": {
            "name": "time_series_wise_msle_mean_sb",
            "goal": "minimize"
        },
    }

    parameters_dict = {
        # -------- Temporal Context --------
        "steps": {"values": [[*range(1, 36 + 1)]]},  # Predict 36 future months
        "input_chunk_length": {"values": [36, 48, 60, 72]},  # >72 adds memory, dilutes sharp pre-spike patterns

        # -------- Training Regime --------
        "batch_size": {"values": [64, 96, 128]},  # Avoid 256+ (averages away rare spike gradients)
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [10]},  # Longer patience for late emerging positive signals

        # -------- Optimization / Scheduler --------
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 8e-4,  # Cap below 1e-3 to reduce rapid zero collapse
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 5e-6,
            "max": 3e-4,  # Excessive WD suppresses small positives
        },
        "lr_scheduler_factor": {
            "distribution": "uniform",
            "min": 0.35,
            "max": 0.6,
        },
        "lr_scheduler_patience": {"values": [3, 5]},
        "lr_scheduler_min_lr": {"values": [1e-6]},

        # -------- Scaling & Logging --------
        "feature_scaler": {"values": ["RobustScaler"]},  # Median/IQR robust to spike outliers
        "target_scaler": {"values": ["RobustScaler"]},   # Single choice for stable threshold semantics
        "log_targets": {"values": [True]},  # log1p expands resolution of 1–5 fatalities
        "log_features": {
            "values": [["lr_ged_sb", "lr_ged_ns", "lr_ged_os", "lr_acled_sb", "lr_acled_os", "lr_ged_sb_tsum_24", 
                         "lr_splag_1_ged_sb", "lr_splag_1_ged_os", "lr_splag_1_ged_ns"]]
        },

        # -------- TFT Architecture --------
        "hidden_size": {"values": [64, 128, 192, 256]},  # Drop 512 (overfits rare spikes)
        "lstm_layers": {"values": [1, 2]},  # Deeper → diminishing returns + noise
        "num_attention_heads": {"values": [2, 4, 8]},  # Ensure divisibility with hidden_size choices
        "full_attention": {"values": [False]},  # Sparse attention keeps memory lower; full not needed
        "feed_forward": {
            "values": [
                "GatedResidualNetwork",  # TFT default; good for sparse gating
                "GEGLU",                 # Smoother gated variant
                "GLU",                   # Simpler gate
                "ReLU"                   # Baseline ablation
            ]
        },
        "attention_dropout": {"values": [0.05, 0.1, 0.2]},  # Avoid 0 (overfit) and >0.3 (loses rare spikes)
        "dropout": {"values": [0.15, 0.25, 0.35]},  # Moderate regularization; >0.4 erodes small signals
        "use_reversible_instance_norm": {"values": [True, False]},  # Can stabilize shifting sparse distributions

        # -------- Loss / Imbalance Handling --------
        "loss_function": {"values": ["WeightedPenaltyHuberLoss"]},
        "zero_threshold": {
            "distribution": "log_uniform_values",
            "min": 3e-4,
            "max": 7e-3,  # Ensures 1–3 fatalities remain non-zero post log+robust scaling
        },
        "false_positive_weight": {
            "distribution": "uniform",
            "min": 1.2,
            "max": 3.0,  # Cap to avoid suppressing emerging conflict signals
        },
        "false_negative_weight": {
            "distribution": "uniform",
            "min": 2.4,
            "max": 5.4,  # Penalize missed spikes; avoid instability > ~6
        },
        "non_zero_weight": {
            "distribution": "uniform",
            "min": 4.0,
            "max": 8.0,  # Maintain gradient from sparse positives
        },
        "delta": {
            "distribution": "log_uniform_values",
            "min": 0.06,
            "max": 1.6,  # Huber transition controls sensitivity to moderate spikes
        },

        # -------- Gradient Stability --------
        "gradient_clip_val": {
            "distribution": "uniform",
            "min": 0.6,
            "max": 1.4,  # Tight window preserves spike gradients, avoids explosions
        },
    }
    sweep_config["parameters"] = parameters_dict

    return sweep_config