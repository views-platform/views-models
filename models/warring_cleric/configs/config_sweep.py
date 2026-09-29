def get_sweep_config():
    """
    """
    sweep_config = {
        "method": "bayes",
        "name": "elastic_heart_tsmixer_shadow_20260508_I",
        "early_terminate": {
            "type": "hyperband",
            # RLROP patience=15 + cooldown=3: first reduction fires at epoch ~18.
            # min_iter=30 ensures at least one LR reduction before Hyperband kills.
            "min_iter": 30,
            "eta": 2,
        },
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
        # mild AdamW shrinkage; LayerNorm self-corrects scale drift. WD=0 removes
        # decoupled regularization entirely, risking per-country memorization.
        "weight_decay": {"values": [1e-3, 1e-4, 5e-5]},
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["ReduceLROnPlateau"]},
        # factor=0.5 halves LR each firing → 3 firings = lr×0.125 (floor hit fast).
        # factor=0.7 reduces 30% each firing → 3 firings = lr×0.343 (3× more LR at floor).
        # factor=0.8 reduces 20% each firing → 3 firings = lr×0.512 (barely reduced).
        # 0.7 is the sweet spot: still meaningful reduction, much more budget per level.
        "lr_scheduler_factor": {"values": [0.5]},
        "lr_scheduler_patience": {"values": [25]},
        "lr_scheduler_min_lr": {"values": [1e-5]},
        "lr_scheduler_kwargs": {"values": [
            {"mode": "min", "factor": 0.5, "patience": 25, "min_lr": 1e-5, "threshold": 0.01, "threshold_mode": "rel", "cooldown": 3},
        ]},
        # clip=[20,50]: grad_norm/max naturally settles ~36 at ep65 with clip=50 → clip never fires.
        # clip=20 provides occasional gradient noise regularization on the hottest batches;
        # clip=50 lets the optimizer run free. Both needed for Bayes to discriminate.
        "gradient_clip_val": {"values": [20.0, 50.0]},
        # ==============================================================================
        # SCALING
        # ==============================================================================
        "feature_scaler": {"values": ["AsinhTransform->MaxAbsScaler"]},  # global chain; covariates derive from the queryset (ADR-013, C-95)
        "target_scaler": {"values": ["AsinhTransform"]},
            
        # ==============================================================================
        # TSMIXER ARCHITECTURE
        # ==============================================================================
        # num_blocks=2 only: 3rd block re-encodes the static country profile (22/31
        # features are annual → identical across the 36-step window). Extra depth adds
        # leakage capacity, not temporal discrimination.
        "num_blocks": {"values": [2]},
        "hidden_size": {"values": [128, 256]},
        # ff_size=256 only: ff=128 with hidden=128 → zero expansion (square projection,
        # monthly and annual features fight for the same 128-dim bottleneck). ff=128
        # with hidden=256 → 0.5× compression, actively destructive. ff=256 gives 2×
        # expansion for hidden=128 and parity for hidden=256 — minimum viable.
        "ff_size": {"values": [256, 512]},
        "normalize_before": {"values": [True]},
        "activation": {"values": ["GELU"]},
        "norm_type": {"values": ["LayerNorm"]},
        
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        # dropout=0.05 removed: ep54→65 shows train_loss −21% while val_loss +3% — memorization.
        # With clip=50 never firing (~36 max), 0.05 leaves the model unregularized against
        # conflict pattern memorization. 0.10 is the new floor; 0.25 retained from sweep C best.
        "dropout": {"values": [0.15, 0.25, 0.35]},
        "use_static_covariates": {"values": [True]},
        "use_reversible_instance_norm": {"values": [True]},
        
        # ==============================================================================
        # STATIC COVARIATE STATS
        # ==============================================================================
        # Per-entity fingerprint stats (mu, sigma, max, trend, sparsity) are
        # injected as static covariates into every TSMixer block via feature_mixing_static.
        # AsinhTransform alone leaves Syria mu≈5.3 vs peaceful countries at 0 — this
        # persistent 5× gap is injected at every block, biasing predictions upward
        # for high-conflict countries and causing systematic overprediction in the
        # 5–50 death range. MaxAbsScaler maps to [0,1]: Syria=1.0, peace=~0,
        # preserving relative order with no structural positive push.
        # Unlike TFT (VSN+GRN can learn to gate/rescale), TSMixer uses blunt linear
        # concatenation — cross-entity scale normalization must be explicit.
        # "static_covariate_stats": {"values": [{"transform": "AsinhTransform->MaxAbsScaler"}]},
        
        # ==============================================================================
        # LOSS FUNCTION: SpotlightLoss
        # ==============================================================================
        "loss_function": {"values": ["SpotlightLossAsinh"]},
        "non_zero_threshold": {"values": [0.88]}, 
        # delta: multi-resolution spectral weight. DC bin masked.
        # "delta": {"distribution": "uniform", "min": 0.0, "max": 0.05},
        "delta": {"values": [-1]},
        
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        # cyclic=False: sin/cos(month) + RevIN mean-strip adds a harmonic bias that
        # the mixer may over-rely on instead of learning conflict patterns.
        # TSMixer has no GRU h_T bottleneck but mixing still routes cyclic signal at every layer.
        "use_cyclic_encoders": {"values": [False, True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config