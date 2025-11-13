import math

def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    This configuration is "operational" so modifying it will change the search strategy, parameter ranges, and other settings for hyperparameter tuning aimed at optimizing model performance.

    Returns:
    - sweep_config (dict): A dictionary containing the configuration for hyperparameter sweeps, defining the methods and parameter ranges used to search for optimal hyperparameters.
    """

    sweep_config = {
        'method': 'bayes',
        'name': 'good_life_transformer_focus',
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 8,      # Give model time to move off trivial zero
            'eta': 2
        },
        'metric': {
            'name': 'time_series_wise_msle_mean_sb',
            'goal': 'minimize'
        },
    }

    parameters_dict = {
        # ---------------- Temporal ----------------
        # Limit maximum context to 72 (84 often adds noise & memory overhead without improving spike capture).
        'steps': {'values': [[*range(1, 36 + 1)]]},
        'input_chunk_length': {'values': [36, 48, 60, 72]},

        # ---------------- Training ----------------
        # Batch size: very large (256+) averages away rare spike gradients; keep moderate range.
        'batch_size': {'values': [64, 96, 128, 192]},
        'n_epochs': {'values': [300]},
        'early_stopping_patience': {'values': [8, 12]},  # Longer patience for late-emerging non-zero prediction behavior

        # ---------------- Optimization ----------------
        # LR upper bound lowered (<1e-3) to avoid rapid convergence to zero baseline; log-uniform for exploration of small rates.
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 7e-4,
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 5e-6,
            'max': 3e-4,  # High WD suppresses small positive signals
        },
        'lr_scheduler_factor': {
            'distribution': 'uniform',
            'min': 0.35,
            'max': 0.6,
        },
        'lr_scheduler_patience': {'values': [3, 5]},
        'lr_scheduler_min_lr': {'values': [1e-6]},

        # ---------------- Scaling & Logging ----------------
        # Single robust target scaler to keep semantics of zero_threshold stable across runs.
        'feature_scaler': {'values': ['RobustScaler']},
        'target_scaler': {'values': ['RobustScaler']},
        'log_targets': {'values': [True]},  # log1p expands low counts without flattening spikes
        'log_features': {
            'values': [["lr_ged_sb", "lr_ged_ns", "lr_ged_os", "lr_acled_sb", "lr_acled_os", "lr_ged_sb_tsum_24", 
                         "lr_splag_1_ged_sb", "lr_splag_1_ged_os", "lr_splag_1_ged_ns"]]
        },

        # ---------------- Transformer Architecture ----------------
        # d_model kept <=256 to prevent memorizing a few historic spikes; 192 included for intermediate capacity.
        'd_model': {'values': [64, 128, 192, 256]},
        # nhead chosen to divide most d_model values cleanly (4 divides all in list; 8 divides 128 & 256).
        'nhead': {'values': [4, 8]},
        # Encoder/decoder depth restricted; deeper stacks overfit sparse tails and inflate memory.
        'num_encoder_layers': {'values': [2, 3]},
        'num_decoder_layers': {'values': [2, 3]},
        # Feedforward dimension roughly 2–3x d_model; exclude 1024 to avoid overfitting spikes.
        'dim_feedforward': {'values': [128, 256, 384, 512]},
        # Activation set includes GELU (smooth), ReLU (baseline), GEGLU (gated), LeakyReLU (better near-zero).
        'activation': {'values': ['GELU', 'ReLU', 'GEGLU', 'LeakyReLU']},
        # Normalization variants; RMSNorm stabilizes scale shifts after log+robust transforms.
        'norm_type': {'values': ['LayerNorm', 'RMSNorm', 'LayerNormNoBias']},
        # Dropout moderate; >0.4 erases rare pattern signals, <0.1 risks overfit to spikes.
        'dropout': {'values': [0.15, 0.25, 0.35]},
        # Gradient clipping tight to retain spike gradients but prevent explosion.
        'gradient_clip_val': {
            'distribution': 'uniform',
            'min': 0.6,
            'max': 1.4,
        },

        # ---------------- Loss & Imbalance Handling ----------------
        'loss_function': {'values': ['WeightedPenaltyHuberLoss']},
        # zero_threshold tuned for post log1p + RobustScaler so 1–3 fatalities remain positive class.
        'zero_threshold': {
            'distribution': 'log_uniform_values',
            'min': 3e-4,
            'max': 7e-3,
        },
        # False positive weight capped to avoid suppression of emerging conflict signals.
        'false_positive_weight': {
            'distribution': 'uniform',
            'min': 1.2,
            'max': 3.2,
        },
        # False negative weight emphasizes catching spikes; range narrowed to prevent instability.
        'false_negative_weight': {
            'distribution': 'uniform',
            'min': 2.5,
            'max': 5.5,
        },
        # Non-zero weight maintains gradient flow from sparse positive months.
        'non_zero_weight': {
            'distribution': 'uniform',
            'min': 4.0,
            'max': 8.0,
        },
        # Huber delta controls sensitivity to moderate spikes; log range balances L2 near small errors / L1 on large.
        'delta': {
            'distribution': 'log_uniform_values',
            'min': 0.06,
            'max': 1.5,
        },
    }
    sweep_config['parameters'] = parameters_dict

    return sweep_config