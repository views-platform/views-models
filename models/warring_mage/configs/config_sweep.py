def get_sweep_config():
    """
    meow
    """
    sweep_config = {
        "method": "bayes",
        "name": "smol_cat_tide_shadow_20260508_A",
        "early_terminate": {"type": "hyperband", "min_iter": 30, "eta": 2},
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        "input_chunk_length": {"values": [36]},
        "output_chunk_shift": {"values": [0]},
        "random_state": {"values": [67]},
        "output_chunk_length": {"values": [36]},
        "optimizer_cls": {"values": ["AdamW"]},
        "mc_dropout": {"values": [False]},
        "num_samples": {"values": [1]},
        "n_jobs": {"values": [-1]},
        # ==============================================================================
        # TRAINING
        # ==============================================================================
        "batch_size": {"values": [128]},
        "n_epochs": {"values": [300]},
        # ESP=35: allows ~4 LR reductions (patience=8 each) before triggering.
        # Each RLROP firing gives the optimizer a reset opportunity; 35 epochs of
        # continuous stagnation despite all reductions is a reliable stop signal.
        # Hyperband (min_iter=15) is the primary fast-kill for clearly bad runs.
        "early_stopping_patience": {"values": [35]},
        "early_stopping_min_delta": {"values": [0.001]},
        "force_reset": {"values": [True]},
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        "lr": {"values": [5e-4, 2e-4, 1e-4]},
        # WD range [1e-4, 5e-5]: LR floor ≈ 5e-4 × 0.5³ = 6e-5. WD=1e-4 is 1.7× floor —
        # mild AdamW shrinkage; LayerNorm + skip path self-corrects scale drift.
        # WD=0 removes decoupled regularization entirely, risking per-country memorization.
        "weight_decay": {"values": [1e-4, 5e-5]},
        # ==============================================================================
        # LR SCHEDULER: ReduceLROnPlateau
        # RLROP on val_loss: val_loss (test partition, frozen scalers) is significantly
        # smoother than train_loss on conflict batches, so RLROP's plateau detection
        # is reliable here. factor=0.5 (halve LR) is gentle enough for a noisy val
        # signal on ~200 series. patience=8: allows ~4 LR drops within the ESP=35
        # window (8, 16, 24, 32 epochs of stagnation) before early stopping triggers —
        # each drop gives the optimizer a fresh shot before committing to stop.
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["ReduceLROnPlateau"]},
        "lr_scheduler_factor": {"values": [0.5]},
        "lr_scheduler_patience": {"values": [8]},
        "lr_scheduler_min_lr": {"values": [1e-6]},
        "lr_scheduler_kwargs": {"values": [{"mode": "min", 
                                            "factor": 0.5, 
                                            "patience": 8, 
                                            "min_lr": 1e-6, 
                                            "threshold": 0.01, 
                                            "threshold_mode": "rel", 
                                            "cooldown": 3}]},
        # TiDE: skip path provides a direct gradient channel (lookback → output)
        # alongside the encoder path. The skip gradient is single-matrix (low norm);
        # encoder gradients spike on conflict timesteps. 2.0–5.0 brackets the expected
        # range — 1.5 was too tight and would clip the encoder's conflict-onset signal.
        # Not pinned: skip vs encoder gradient balance varies with hidden_size.
        "gradient_clip_val": {"values": [2.0, 3.0, 5.0]},
        # ==============================================================================
        # SCALING
        # ==============================================================================
        "feature_scaler": {"values": ["AsinhTransform->MaxAbsScaler"]},  # global chain; covariates derive from the queryset (ADR-013, C-95)
        "target_scaler": {"values": ["AsinhTransform"]},
        # ==============================================================================
        # TiDE ARCHITECTURE
        # ==============================================================================
        "num_encoder_layers": {"values": [2, 3]},
        # num_decoder_layers=1: single projection from hidden to per-step output.
        # Avoids step-specific memorization of conflict patterns across 36 steps.
        # 2 layers adds capacity to model escalation/de-escalation profiles.
        "num_decoder_layers": {"values": [1, 2]},
        # decoder_output_dim: per-step bottleneck before projecting to 1 value.
        # Tighter bottleneck (16) forces compact representation — prevents the decoder
        # from allocating dedicated dimensions to rare-conflict steps.
        "decoder_output_dim": {"values": [16, 32]},
        "hidden_size": {"values": [64, 128, 256]},
        # forces covariate projection to select conflict-risk indicators over noise.
        "temporal_width_past": {"values": [16, 24]},
        "temporal_width_future": {"values": [4, 6]},
        "temporal_decoder_hidden": {"values": [128, 256]},
        "temporal_hidden_size_past": {"values": [64]},
        "temporal_hidden_size_future": {"values": [32]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        "use_layer_norm": {"values": [True]},
        # Dropout: Country-level has fewer training windows per series.
        # Slightly higher dropout ceiling to prevent overfitting on ~200 series.
        # dropout: TiDE has encoder + decoder + temporal decoder = more parameter paths
        # than TSMixer. Higher dropout (0.35) prevents each path from specialising to
        # event-series memorization. 0.15 preserves conflict-onset gradients in the
        # encoder but risks overfitting on ~13 event entities.
        "dropout": {"values": [0.15, 0.25, 0.35]},
        "use_static_covariates": {"values": [True]},
        # RevIN on: SpotlightLoss DC/AC decomposition zeroes out per-series shape
        # gradients (Σ ∂L_shape/∂ŷᵢ = 0), preventing DC offset amplification through
        # RevIN denormalisation ŷ = ẑ·σ + μ. Safe even for sparse peace series.
        "use_reversible_instance_norm": {"values": [True]},
        "loss_function": {"values": ["SpotlightLossLogcosh"]},
        "non_zero_threshold": {"values": [0.88]}, 
        # delta: multi-resolution spectral weight. DC bin masked.
        # Cap at 0.05: at delta>0.05 on sparse conflict data the model hallucinates
        # broadband noise to reduce spectral loss, raising peace_mean and MSLE.
        # "delta": {"distribution": "uniform", "min": 0.0, "max": 0.05},
        # "static_covariate_stats": {"values": [{"transform": "AsinhTransform"}]},
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config