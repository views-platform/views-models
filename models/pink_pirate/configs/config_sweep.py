def get_sweep_config():
    """
    Learnable vs fixed per-target sigma sweep (ADR-055).

    The per-target sweep confirmed {sb: 1.0, ns: 0.75, os: 0.5} as the
    best fixed combination. This sweep tests whether learnable sigma
    (optimizer-tuned) improves on the hand-picked values, and whether
    the initialization point matters.

    Sweep axes:
        learnable_sigma — True / False
        loss_reg_sigma  — 3 initialization points

    Total runs: 6 (2 × 3 grid)
    Lessons: 80
    """

    sweep_config = {
        "name": "pink_pirate_learnable_sigma_sweep",
        "method": "grid",
    }

    metric = {
        "name": "36month_mean_squared_error",
        "goal": "minimize",
    }

    sweep_config["metric"] = metric

    parameters_dict = {
        # ============================================================
        # Ledger / Topology (ADR 007 Compliance)
        # ============================================================
        "time_col": {"value": "month_id"},
        "id_col": {"value": "priogrid_gid"},
        "spatial_cols": {"value": ["row", "col"]},
        "identity_cols": {"value": ["month_id", "priogrid_gid", "c_id", "row", "col"]},
        "index_names": {"value": ["month_id", "priogrid_gid"]},
        "features": {"value": ["lr_sb_best", "lr_ns_best", "lr_os_best"]},
        "input_channels": {"value": 3},
        "row_offset": {"value": 87},
        "col_offset": {"value": 310},
        "height": {"value": 180},
        "width": {"value": 180},
        # ============================================================
        # Model Architecture
        # ============================================================
        "model": {"value": "HydraBNUNet06_LSTM4"},
        "total_hidden_channels": {"value": 32},
        "dropout_rate": {"value": 0.125},
        "window_dim": {"value": 32},
        "output_channels": {"value": 1},
        "weight_init": {"value": "xavier_norm"},
        "h_init": {"value": "abs_rand_exp-100"},
        # ============================================================
        # Optimization (ADR 014 Compliance)
        # ============================================================
        "windows_per_lesson": {"value": 3},
        "learning_rate": {"value": 0.001},
        "weight_decay": {"value": 0.1},
        "scheduler": {"value": "WarmupDecay"},
        "warmup_steps": {"value": 100},
        "clip_grad_norm": {"value": True},
        "torch_seed": {"value": 4},
        "np_seed": {"value": 4},
        # ============================================================
        # Multi-Task Signals (ADR 020 Compliance)
        # ============================================================
        "classification_targets": {
            "value": ["by_sb_best", "by_ns_best", "by_os_best"],
        },
        "regression_targets": {
            "value": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
        },
        "transformations": {
            "value": {
                "log1p": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
                "asinh": [],
                "identity": [],
            },
        },
        "derivations": {
            "value": {
                "binary": [
                    {"from": "lr_sb_best", "to": "by_sb_best", "threshold": 0},
                    {"from": "lr_ns_best", "to": "by_ns_best", "threshold": 0},
                    {"from": "lr_os_best", "to": "by_os_best", "threshold": 0},
                ],
            },
        },
        "steps": {"value": list(range(1, 37))},
        "time_steps": {"value": 36},
        # ============================================================
        # Loss Functions — Tobit (ADR-054)
        # ============================================================
        "loss_reg": {"value": "tobit"},
        "loss_class": {"value": "focal"},
        "loss_class_alpha": {"value": 0.75},
        "loss_class_gamma": {"value": 1.5},
        "onset_bias_init": {"value": -7.0},
        # ============================================================
        # SWEEP AXIS 1: Learnable sigma (ADR-055)
        # ============================================================
        "learnable_sigma": {
            "values": [False, True],
        },
        # ============================================================
        # SWEEP AXIS 2: Per-target sigma initialization
        #
        #   1. Proposed optimum from sweep (the hand-picked best)
        #   2. Uniform 1.0 (does the optimizer find per-target values
        #      from a naive start?)
        #   3. Wider spread (does aggressive init help or hurt?)
        # ============================================================
        "loss_reg_sigma": {
            "values": [
                {"lr_sb_best": 1.0, "lr_ns_best": 0.75, "lr_os_best": 0.5},
                {"lr_sb_best": 1.0, "lr_ns_best": 1.0, "lr_os_best": 1.0},
                {"lr_sb_best": 1.5, "lr_ns_best": 0.75, "lr_os_best": 0.25},
            ],
        },
        # ============================================================
        # Strategy (Curriculum ADR 011/012 Compliance)
        # ============================================================
        "total_lessons": {"value": 80},
        "max_ratio": {"value": 0.95},
        "min_ratio": {"value": 0.05},
        "slope_ratio": {"value": 0.75},
        "roof_ratio": {"value": 0.7},
        "min_events": {"value": 5},
        "sampling_strategy": {"value": "threshold"},
        # ============================================================
        # Outbound / Evaluation
        # ============================================================
        "n_posterior_samples": {"value": 3},
        "evaluation_mode": {"value": "stochastic"},
        "aggregate_method": {"value": "arithmetic_mean"},
        "skip_predictions_delivery": {"value": True},
    }

    sweep_config["parameters"] = parameters_dict

    return sweep_config
