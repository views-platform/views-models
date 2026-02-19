def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.

    NOTE: This configuration was restored from a previous best-performing run
    (MSLE ~0.40 from wandb run `3j9x0s4r`) to establish a replicable baseline.
    """

    hyperparameters = {
        # --- Forecast horizon ---
        "steps": list(range(1, 37)),

        # --- Architecture ---
        "activation": "LeakyReLU",
        "generic_architecture": True,
        "num_stacks": 2,
        "num_blocks": 3,
        "num_layers": 3,
        "layer_widths": 64, # widths 
        "dropout": 0.3,
        "batch_norm": False,          

        # --- Input / output structure ---
        "input_chunk_length": 24,
        "output_chunk_length": 36,
        "output_chunk_shift": 0,

        # --- Training ---
        "batch_size": 8,
        "n_epochs": 100,
        "early_stopping_patience": 1, #40
        "early_stopping_min_delta": 0.01,
        "gradient_clip_val": 1.0,
        "force_reset": True,
        "random_state": 1,            # selected from sweep [1, 2]

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
        "feature_scaler": None, #"MinMaxScaler",
        "target_scaler": None, # "MinMaxScaler",
        "log_targets": None,
        "log_features": None,
        "use_reversible_instance_norm": False, # True, # False, # darts native
        # "use_static_covariates": True, 

        # --- Loss & penalties ---
        "loss_function": "TweedieLoss",
        "p": 1.5,
        "eps": 1e-6,
        "zero_threshold": 0.05,
        "non_zero_weight": 7.0,
        "false_positive_weight": 1.0,
        "false_negative_weight": 10.0,

        # --- Probabilistic / sampling ---
        "num_samples": 1,
        "mc_dropout": False, #True,

        # --- other ---
        "n_jobs": -1
    }

    return hyperparameters
