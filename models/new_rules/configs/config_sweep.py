def get_sweep_config():
    """
    meow
    """
    sweep_config = {
        "method": "bayes",
        "name": "new_rules_nbeats_prism_v14_msle",
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 25,
            "eta": 2,
        },  # 50 > CAWR T_0=30 — avoids pruning at restart spike edge
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        # icl=48: 4yr context. N-BEATS flattens the full input window to one vector,
        # so larger icl increases the non-zero fraction of that vector and gives the
        # FC layers more conflict signal to compress. icl=36 gives no lookback
        # advantage over the output horizon — conflict persistence requires more context.
        "input_chunk_length": {"values": [36]},
        "output_chunk_length": {"values": [36]},
        "output_chunk_shift": {"values": [0]},
        "random_state": {"values": [67]},
        "mc_dropout": {"values": [False]},
        "optimizer_cls": {"values": ["AdamW"]},
        "num_samples": {"values": [1]},
        "n_jobs": {"values": [-1]},
        # ==============================================================================
        # TRAINING
        # ==============================================================================
        "batch_size": {"values": [256]},
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [30]},
        "early_stopping_min_delta": {"values": [0.0001]},
        "force_reset": {"values": [True]},
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        "lr": {
            "distribution": "log_uniform_values",
            "min": 3e-4,
            "max": 1e-3,
        },
        "weight_decay": {"values": [0, 1e-4, 1e-3]},
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["ReduceLROnPlateau"]},
        "lr_scheduler_factor": {"values": [0.5]},
        "lr_scheduler_patience": {"values": [15]},
        "lr_scheduler_min_lr": {"values": [1e-6]},
        "gradient_clip_val": {"values": [1.0]},
        # ==============================================================================
        # SCALING
        # ==============================================================================
        "feature_scaler": {"values": [None]},
        "target_scaler": {"values": ["LogTransform"]},  # log1p(x): model operates directly in MSLE space
        "feature_scaler_map": {
            "values": [
                {
                    "LogTransform": [
                        "lr_wdi_sp_dyn_imrt_fe_in",
                        "lr_wdi_sm_pop_refg_or",
                        "lr_wdi_ny_gdp_mktp_kd",
                        "lr_wdi_nv_agr_totl_kn",
                        "lr_splag_1_ged_sb",
                        "lr_splag_1_ged_ns",
                        "lr_splag_1_ged_os",
                        "lr_ged_ns", 
                        "lr_ged_os",
                    ],
                    "StandardScaler": [
                        "lr_ged_sb_delta",
                        "lr_ged_ns_delta",
                        "lr_ged_os_delta",
                        "lr_wdi_sm_pop_netm",
                        "lr_wdi_dt_oda_odat_pc_zs",
                        "lr_wdi_sp_pop_grow",
                        "lr_wdi_ms_mil_xpnd_gd_zs",
                    ],
                    "MinMaxScaler": [
                        "lr_wdi_sh_sta_stnt_zs",
                        "lr_wdi_sh_sta_maln_zs",
                        "lr_wdi_sl_tlf_totl_fe_zs",
                        "lr_wdi_se_enr_prim_fm_zs",
                        "lr_wdi_sp_urb_totl_in_zs",
                        "lr_vdem_v2x_horacc",
                        "lr_vdem_v2x_veracc",
                        "lr_vdem_v2x_diagacc",
                        "lr_vdem_v2xnp_client",
                        "lr_vdem_v2xnp_regcorr",
                        "lr_vdem_v2xpe_exlpol",
                        "lr_vdem_v2xpe_exlgeo",
                        "lr_vdem_v2xpe_exlgender",
                        "lr_vdem_v2xpe_exlsocgr",
                        "lr_vdem_v2x_divparctrl",
                        "lr_vdem_v2x_ex_party",
                        "lr_vdem_v2x_ex_military",
                        "lr_vdem_v2x_genpp",
                        "lr_vdem_v2xeg_eqdr",
                        "lr_vdem_v2xcl_prpty",
                        "lr_vdem_v2xeg_eqprotec",
                        "lr_vdem_v2xcl_dmove",
                        "lr_vdem_v2x_clphy",
                    ],
                }
            ]
        },
        # ==============================================================================
        # N-BEATS ARCHITECTURE
        # ==============================================================================
        # generic_architecture: True uses generic basis (learnable), False
        # uses interpretable trend+seasonality decomposition. Generic is
        # more flexible for conflict data which lacks clean seasonality.
        "generic_architecture": {"values": [True]},
        # num_stacks: Number of stacks. Each stack processes the residual
        # from the previous. 2 is standard for generic, more adds capacity.
        # num_stacks: Fixed at 2. 3 adds ~50% params with marginal gain for
        # conflict data — N-BEATS residual path saturates fast. Saves search space.
        "num_stacks": {"values": [2]},
        # num_blocks: Blocks per stack. N-BEATS paper uses 1 per stack for generic.
        # Keep low — each additional block adds a backcast path; the final block's
        # backcast is structurally discarded, and with 4 blocks per stack the
        # dead-backcast cascade can propagate backward. 1-2 avoids this.
        "num_blocks": {"values": [1, 2]},
        # num_layers: FC layers per block. 2-4 is standard. Deeper blocks
        # capture more complex patterns but risk overfitting on ~200 series.
        # num_layers: Fixed at 3. Paper uses 4; 2 is capacity-limited for
        # conflict dynamics. 3 is the right balance.
        "num_layers": {"values": [3, 4]},
        # layer_widths: Width of FC layers in each block. N-BEATS flattens
        # input_chunk_length * n_features into a single vector (~48×40≈1920
        # dims). layer_widths=128 is an ~15:1 compression — sparse inputs mean
        # the conflict signal (5% of cells) gets averaged out at that bottleneck.
        # 256-512 keeps the compression ratio manageable (~4-8×) and preserves
        # peak values instead of pulling predictions toward the zero mean.
        # layer_widths: 128 dropped — confirmed capacity-limited floor in all
        # sweep runs. 256 is minimum viable, 512 gives headroom for diverse
        # country trajectories.
        "layer_widths": {"values": [256, 512]},
        # expansion_coefficient_dim: Dimensionality of basis expansion
        # coefficients (generic mode). Must be ≤ ocl=36 to avoid overcomplete
        # basis (null space → OOD explosions). ed=16 covers 44% of R^36;
        # ed=5 is an aggressive regularizer. Both well-conditioned.
        "expansion_coefficient_dim": {"values": [5, 16]},
        # trend_polynomial_degree: Only used in interpretable mode.
        # Included for completeness; irrelevant when generic=True.
        "trend_polynomial_degree": {"values": [2]},
        # activation: ReLU is N-BEATS paper default.
        "activation": {"values": ["ReLU"]},
        # use_reversible_instance_norm: Fixed False for this sweep.
        # N-BEATS has no internal normalization in its residual path — pure FC →
        # basis expansion. With dual_mean=False the upward gradient bias is gone,
        # so RevIN=True no longer amplifies a DC offset. But without RevIN=False
        # confirmed safe first, we keep it off. Re-introduce in v11 if v10 is clean.
        "use_reversible_instance_norm": {"values": [True]},
        "use_static_covariates": {"values": [True]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        # Dropout: N-BEATS is a deep MLP — moderate dropout needed for
        # ~200 series. Paper uses 0.0 but they had much more data.
        # Dropout: 0.10 is too low for ~200 series (paper used much more data).
        "dropout": {"values": [0.15, 0.25]},
        # ==============================================================================
        # LOSS FUNCTION: PrismLoss
        # ==============================================================================
        "loss_function": {"values": ["PrismLoss"]},
        # alpha removed in PrismLoss v33. MSE in log1p = MSLE exactly.
        # log_cosh+alpha was causing 10-16× gradient deficit via tanh saturation.
        "non_zero_threshold": {"values": [0.693]},  # log1p(1) ≈ 0.693, i.e. ≥1 battle-related death
        # ── delta (multi-resolution spectral weight) ─────────────────────────────────
        # Spectral log_cosh(|S_pred| - |S_true|) at n_fft=6,12,24. DC bin masked.
        # MSE pointwise gradient scales as e (up to ~11). Spectral bounded at tanh ≤ 1.
        # Floor at 0.05: below this spectral is noise vs MSE signal.
        "delta": {
            "distribution": "uniform",
            "min": 0.10,
            "max": 0.25,
        },
        # dual_mean=False: plain per-cell mean = training loss is MSLE exactly.
        # True caused false minimum (train_loss=0.28 vs MSLE_val=0.80).
        "dual_mean": {"values": [False]},
        "event_weight": {
            "values": [0.50], # not used
        },
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config
