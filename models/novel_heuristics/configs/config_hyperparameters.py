def get_hp_config():
    """
    Hyperparameter configuration transferred from W&B sweep `novel_heuristics_04`.

    NOTE:
    - This is a controlled transfer from the sweep config.
    - All stochasticity and random seed behavior is preserved (no seed added or modified).
    - Differences vs. the old baseline are intentional and documented.
    """

    hyperparameters = {
        # --- Forecast horizon ---
        "steps": list(range(1, 37)),

        # --- Architecture ---
        "activation": "LeakyReLU",
        "generic_architecture": True,
        "num_stacks": 2,
        "num_blocks": 4,            # UPDATED (was 3)
        "num_layers": 2,            # UPDATED (was 1)
        "layer_widths": 16,
        "dropout": 0.3,
        "batch_norm": False,
        "mc_dropout": True,

        # --- Input / output structure ---
        "input_chunk_length": 24,
        "output_chunk_length": 36,
        "output_chunk_shift": 0,

        # --- Training ---
        "batch_size": 8,
        "n_epochs": 300,
        "early_stopping_patience": 20,
        "early_stopping_min_delta": 0.001,
        "gradient_clip_val": 0.64,  # UPDATED
        "force_reset": True,

        # --- Optimizer ---
        "lr": 0.0006,               # UPDATED
        "weight_decay": 0.0003,     # UPDATED
        "optimizer_cls": "Adam",
        "optimizer_kwargs": {
            "lr": 0.0006,
            "weight_decay": 0.0003,
        },

        # --- LR scheduler ---
        "lr_scheduler_cls": "ReduceLROnPlateau",
        "lr_scheduler_factor": 0.46,   # UPDATED
        "lr_scheduler_min_lr": 0.00001,
        "lr_scheduler_patience": 7,
        "lr_scheduler_kwargs": {
            "mode": "min",
            "factor": 0.46,
            "min_lr": 0.00001,
            "monitor": "train_loss",
            "patience": 7,
        },

        # --- Scaling & transforms ---
        "feature_scaler": "MinMaxScaler",
        "target_scaler": None,      # UPDATED (was MinMaxScaler)
        "log_targets": True,
        "log_features": None,       # UPDATED (was explicit feature list)

        # --- Loss & penalties ---
        "loss_function": "WeightedPenaltyHuberLoss",
        "delta": 0.13,              # UPDATED
        "zero_threshold": 0.13,     # UPDATED
        "non_zero_weight": 2.5,
        "false_negative_weight": 4.0,   # UPDATED
        "false_positive_weight": 1.5,   # UPDATED

        # --- Probabilistic / sampling ---
        "num_samples": 1,

        # --- Model plumbing (unchanged, out of sweep scope) ---
        "input_dim": 72,
        "output_dim": 1,
        "nr_params": 1,
        "likelihood": None,
        "train_sample_shape": [
            [24, 1],
            [24, 71],
            None,
            None,
            None,
            [36, 1],
        ],
        "trend_polynomial_degree": 2,
        "expansion_coefficient_dim": 5,
        "use_reversible_instance_norm": False,
    }

    return hyperparameters
