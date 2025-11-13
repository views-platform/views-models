def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    This configuration is "operational" so modifying it will change the search strategy, parameter ranges, and other settings for hyperparameter tuning aimed at optimizing model performance.

    Returns:
    - sweep_config (dict): A dictionary containing the configuration for hyperparameter sweeps, defining the methods and parameter ranges used to search for optimal hyperparameters.
    """

    sweep_config = {
        "method": "bayes",
        "name": "new_rules_nbeats_focus",
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 8,  # Allow emergence of non-zero predictions
            "eta": 2
        },
        "metric": {
            "name": "time_series_wise_msle_mean_sb",
            "goal": "minimize"
        },
    }

    parameters_dict = {
        # -------- Temporal Context --------
        # Longer lookbacks help capture escalation patterns; >84 rarely adds benefit vs memory cost.
        "steps": {"values": [[*range(1, 36 + 1)]]},
        "input_chunk_length": {"values": [36, 48, 60, 72, 84]},

        # -------- Training Regime --------
        # Moderate batch sizes: very large (256+) can dilute rare spike gradients.
        "batch_size": {"values": [64, 128, 192]},
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [8, 12]},  # Give time for late small positives to emerge

        # -------- Optimization / Scheduler --------
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 7e-4,  # Cap below 1e-3 to prevent rapid zero collapse
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 5e-6,
            "max": 4e-4,  # Excessive WD suppresses small positives
        },
        "lr_scheduler_factor": {
            "distribution": "uniform",
            "min": 0.35,
            "max": 0.6,
        },
        "lr_scheduler_patience": {"values": [3, 5]},
        "lr_scheduler_min_lr": {"values": [1e-6]},

        # -------- Scaling & Logging --------
        # Single scaler for consistency with zero_threshold semantics.
        "feature_scaler": {"values": ["RobustScaler"]},
        "target_scaler": {"values": ["RobustScaler"]},
        "log_targets": {"values": [True]},  # log1p expands 1–5 fatalities
        "log_features": {
            "values": [["lr_ged_sb", "lr_ged_ns", "lr_ged_os", "lr_acled_sb", "lr_acled_os",
                        "lr_ged_sb_tsum_24", "lr_splag_1_ged_sb", "lr_splag_1_ged_os", "lr_splag_1_ged_ns"]]
        },

        # -------- N-BEATS Architecture --------
        # generic_architecture False enables interpretable stacks; both tested.
        "generic_architecture": {"values": [True, False]},
        # Stacks & blocks control hierarchical decomposition; avoid very deep (>7) to reduce overfit to rare spikes.
        "num_stacks": {"values": [3, 4, 5, 6]},
        "num_blocks": {"values": [1, 2, 3]},
        # Layers per block: shallow (1–3) keeps focus on sharp changes rather than smooth long trends only.
        "num_layers": {"values": [2, 3]},
        # Layer widths: cap at 512 (1024 often memorizes extreme events); include smaller for regularization.
        "layer_widths": {"values": [128, 256, 384, 512]},
        # Dropout moderate to retain rare spikes; >0.4 erodes small escalation signals.
        "dropout": {"values": [0.1, 0.2, 0.3, 0.35]},
        # Activation variety to handle sparse gradients; LeakyReLU prevents dead units; GELU smooth; ELU aids near-zero.
        "activation": {"values": ["ReLU", "LeakyReLU", "GELU", "ELU"]},

        # -------- Loss / Imbalance Handling --------
        "loss_function": {"values": ["WeightedPenaltyHuberLoss"]},
        # zero_threshold tuned post log+RobustScaler: keeps 1–3 fatalities > threshold (not collapsed to zero).
        "zero_threshold": {
            "distribution": "log_uniform_values",
            "min": 3e-4,
            "max": 7e-3,
        },
        # FP penalty capped to avoid suppressing emerging conflict predictions.
        "false_positive_weight": {
            "distribution": "uniform",
            "min": 1.2,
            "max": 3.2,
        },
        # FN penalty emphasizes missed spikes; range restrained to prevent instability.
        "false_negative_weight": {
            "distribution": "uniform",
            "min": 2.5,
            "max": 5.5,
        },
        # Non-zero weight maintains gradient signal from sparse positive months.
        "non_zero_weight": {
            "distribution": "uniform",
            "min": 4.0,
            "max": 8.5,
        },
        # Huber delta controls transition; smaller values increase sensitivity to modest spikes.
        "delta": {
            "distribution": "log_uniform_values",
            "min": 0.06,
            "max": 1.6,
        },

        # -------- Gradient Stability --------
        "gradient_clip_val": {
            "distribution": "uniform",
            "min": 0.6,
            "max": 1.4,  # Tight window protects against spike-driven exploding gradients
        },
    }

    sweep_config["parameters"] = parameters_dict

    return sweep_config