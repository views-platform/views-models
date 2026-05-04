
def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.
    This configuration is "operational" so modifying these settings will impact the model's behavior during the training.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for training the model, which determine the model's behavior during the training phase.
    """
    
    hyperparameters = {
        'steps': [*range(1, 36 + 1, 1)],

        "use_cyclic_encoders": True,
        "time_steps": 36,  # Checksum: Must match len(steps)
        "rolling_origin_stride": 1,
        "n_jobs": -1,
        "use_cyclic_encoders": True,

        # Prediction output format
        "prediction_format": "dataframe",
        "static_covariate_stats": {"transform": "AsinhTransform->MaxAbsScaler"},

        "num_samples": 1,
        "mc_dropout": False,

        # gradient_clip_val=1.0: was 5.0, which allowed the model to learn
        # ẑ≈9 in normalized space for OOD high-conflict inputs (e.g. Ukraine).
        # RevIN denorm (×log1p(σ)+μ) + sinh then amplifies this to billions.
        # clip=1.0 keeps weight norms conservative so OOD inputs can't produce
        # extreme activations through 5 dilated conv layers.
        "gradient_clip_val": 1.0,

        # CAWR T_mult=2: exponentially growing cycles (25→50→100→200 epochs).
        # T_mult=1 fires 12 LR restarts in 300 epochs; each spike temporarily
        # overwhelms weight_decay on the weight_norm g parameter → g grows →
        # RevIN denorm + sinh blowup. T_mult=2 gives only 3 restarts total.
        "lr_scheduler_T_mult": 2,

        # early_stopping_patience must exceed the longest early cycle.
        # With T_mult=2 and T_0=25, cycle 2 spans epochs 25–75 (50 epochs),
        # cycle 3 spans 75–175 (100 epochs). Patience < 100 can fire mid-cycle
        # during the warmup trough and kill a valid training run prematurely.
        "early_stopping_patience": 100,
    }
    return hyperparameters
