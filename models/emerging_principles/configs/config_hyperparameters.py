
def get_hp_config():
    """
    Hyperparameter configuration transferred from W&B sweep
    `emerging_principles_26`.

    N-BEATS model with 2 stacks and 4 blocks per stack. Each block uses
    3 fully connected layers of width 64, with LeakyReLU activation and
    a dropout rate of 0.3. The model uses a 36-month lookback window
    (input chunk length) and forecasts 36 months ahead.

    NOTE:
    - This is a controlled, deterministic transfer from the sweep definition.
    - All hyperparameters correspond to concrete sweep values.
    - One random seed (`random_state = 1`) is selected to freeze the configuration.   - Deterministic post-sweep configuration (random_state = 1).
    - No added stochasticity beyond MC dropout.
    """

    hyperparameters = {
        # --- Forecast horizon ---
        "steps": list(range(1, 37)),

        # --- Architecture ---
        "activation": "LeakyReLU",
        "generic_architecture": True,
        "num_stacks": 2,
        "num_blocks": 4,
        "num_layers": 3,
        "layer_widths": 64,
        "dropout": 0.3,
        "batch_norm": False,
        "mc_dropout": True,

        # --- Input / output structure ---
        "input_chunk_length": 36,
        "output_chunk_length": 36,
        "output_chunk_shift": 0,

        # --- Training ---
        "batch_size": 8,
        "n_epochs": 300,
        "early_stopping_patience": 40,
        "early_stopping_min_delta": 0.01,
        "gradient_clip_val": 1.0,
        "force_reset": True,
        "random_state": 1,
        "n_jobs": -1,

        # --- Optimizer ---
        "lr": 0.0003,
        "weight_decay": 0.0003,
        "optimizer_cls": "Adam",

        # --- LR scheduler ---
        "lr_scheduler_cls": "ReduceLROnPlateau",
        "lr_scheduler_factor": 0.46,
        "lr_scheduler_min_lr": 0.00001,
        "lr_scheduler_patience": 7,

        # --- Scaling & transforms ---
        "feature_scaler": "MinMaxScaler",
        "target_scaler": "MinMaxScaler",
        "log_targets": True,
        "log_features": None,
        "use_reversible_instance_norm": False,

        # --- Loss & penalties ---
        "loss_function": "WeightedPenaltyHuberLoss",
        "delta": 0.025,
        "zero_threshold": 0.01,
        "non_zero_weight": 7.0,
        "false_positive_weight": 1.0,
        "false_negative_weight": 10.0,

        # --- Probabilistic / sampling ---
        "num_samples": 1,
    }

    return hyperparameters

