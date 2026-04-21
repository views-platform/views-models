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

        "random_state": 67,
        "time_steps": 36,  # Checksum: Must match len(steps)
        "rolling_origin_stride": 1,
        "prediction_format": "dataframe",

        # --- other ---
        "n_jobs": -1
    }

    return hyperparameters
