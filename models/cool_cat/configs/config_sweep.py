def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    This configuration is "operational" so modifying it will change the search strategy, parameter ranges, and other settings for hyperparameter tuning aimed at optimizing model performance.

    Returns:
    - sweep_config (dict): A dictionary containing the configuration for hyperparameter sweeps, defining the methods and parameter ranges used to search for optimal hyperparameters.
    """

    sweep_config = {
        'method': 'bayes',
        'name': 'cool_cat_tide_focus',
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 10,  # allow emergence of non-zero predictions
            'eta': 2
        },
        'metric': {
            'name': 'time_series_wise_msle_mean_sb', 
            'goal': 'minimize'
        },
    }

    parameters_dict = {
        # ---------------- Temporal ----------------
        # History window: longer than 24 to capture pre-escalation patterns, capped at 60 to avoid dilution.
        'steps': {'values': [[*range(1, 36 + 1)]]},
        'input_chunk_length': {'values': [36, 48, 60]},

        # ---------------- Training ----------------
        'batch_size': {'values': [64, 96, 128]},  # Balanced gradient diversity vs stability
        'n_epochs': {'values': [300]},
        'early_stopping_patience': {'values': [8, 12]},  # Give time for rare positives to appear

        # ---------------- Optimization ----------------
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 1.2e-5,
            'max': 8e-4,  # Upper bound below 1e-3 to reduce rapid collapse to zeros
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 5e-6,
            'max': 2e-4,  # Moderate regularization; high WD suppresses small signals
        },
        'lr_scheduler_factor': {
            'distribution': 'uniform',
            'min': 0.35,
            'max': 0.6,
        },
        'lr_scheduler_patience': {'values': [3, 5]},
        'lr_scheduler_min_lr': {'values': [1e-6]},

        # ---------------- Scaling & Logging ----------------
        # RobustScaler preferred for conflict targets; single scaler to ensure consistent zero_threshold semantics.
        'feature_scaler': {
            'values': ['RobustScaler']
        },
        'target_scaler': {
            'values': ['RobustScaler']
        },
        'log_targets': {'values': [True]},  # log1p before scaling to expand low-count resolution
        'log_features': {
            'values': [["lr_ged_sb", "lr_ged_ns", "lr_ged_os", "lr_acled_sb", "lr_acled_os", "lr_ged_sb_tsum_24", 
                         "lr_splag_1_ged_sb", "lr_splag_1_ged_os", "lr_splag_1_ged_ns"]]
        },

        # ---------------- TiDE Architecture ----------------
        # Shallow layers (1–2) per paper & to avoid overfit to rare spikes.
        'num_encoder_layers': {'values': [1, 2]},
        'num_decoder_layers': {'values': [1, 2]},
        # Hidden capacity: keep upper bound at 256; 512 frequently memorizes spikes.
        'hidden_size': {'values': [64, 128, 192, 256]},
        # Decoder output dim smaller than hidden_size to constrain final projection noise.
        'decoder_output_dim': {'values': [16, 32, 48, 64]},
        # Temporal widths: small receptive fields retain sharp escalation transitions.
        'temporal_width_past': {'values': [2, 4, 6]},
        'temporal_width_future': {'values': [2, 4, 6]},
        # Temporal decoder hidden size: moderate range; large sizes over-smooth spikes.
        'temporal_decoder_hidden': {'values': [32, 48, 64]},
        'use_layer_norm': {'values': [True, False]},
        'dropout': {'values': [0.15, 0.25, 0.35]},  # High dropout (>0.4) erases rare spike signal
        'gradient_clip_val': {
            'distribution': 'uniform',
            'min': 0.5,
            'max': 1.4,  # Tight clipping keeps spike gradients usable but prevents explosions
        },

        # ---------------- Loss (Spike Sensitivity) ----------------
        'loss_function': {'values': ['WeightedPenaltyHuberLoss']},
        # Single zero_threshold range (post log + robust scaling). Ensures 1–3 fatalities not merged into zero.
        'zero_threshold': {
            'distribution': 'log_uniform_values',
            'min': 3e-4,
            'max': 7e-3,
        },
        # false_positive_weight: keep low upper bound; high penalties suppress early escalation predictions.
        'false_positive_weight': {
            'distribution': 'uniform',
            'min': 1.2,
            'max': 3.0,
        },
        # false_negative_weight: emphasize missing spikes; too large destabilizes gradients.
        'false_negative_weight': {
            'distribution': 'uniform',
            'min': 2.4,
            'max': 5.4,
        },
        # non_zero_weight: reinforces learning of rare conflict months; avoid very high (>9) to prevent over-correction.
        'non_zero_weight': {
            'distribution': 'uniform',
            'min': 4.0,
            'max': 8.0,
        },
        # Huber delta: lower bound preserves sensitivity to small deviations; upper bound avoids masking moderate spikes.
        'delta': {
            'distribution': 'log_uniform_values',
            'min': 0.06,
            'max': 1.6,
        },
    }
    sweep_config['parameters'] = parameters_dict

    return sweep_config