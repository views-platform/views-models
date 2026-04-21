def get_sweep_config():
    """
    meow
    """
    sweep_config = {
        "method": "bayes",
        "name": "new_rules_nbeats_spotlight_v3_msle",
        "early_terminate": {"type": "hyperband", "min_iter": 30, "eta": 2},
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        "input_chunk_length": {"values": [48]},
        "output_chunk_length": {"values": [36]},
        "output_chunk_shift": {"values": [0]},
        "random_state": {"values": [67]},
        "mc_dropout": {"values": [False]},
        "optimizer_cls": {"values": ["AdamW", "RAdam"]},
        "num_samples": {"values": [1]},
        "n_jobs": {"values": [-1]},
        # ==============================================================================
        # TRAINING
        # ==============================================================================
        "batch_size": {"values": [64]},
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [50]},
        "early_stopping_min_delta": {"values": [0.0001]},
        "force_reset": {"values": [True]},
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        "lr": {
            "distribution": "log_uniform_values",
            "min": 5e-5,
            "max": 1e-3,
        },
        "weight_decay": {"values": [0, 1e-5, 1e-4]},
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["CosineAnnealingWarmRestarts"]},
        "lr_scheduler_T_0": {"values": [30]},
        "lr_scheduler_T_mult": {"values": [2]},
        "lr_scheduler_eta_min": {"values": [0.0]},
        "gradient_clip_val": {"values": [2.0, 3.0]},
        # ==============================================================================
        # SCALING
        # ==============================================================================
        "feature_scaler": {"values": [None]},
        "target_scaler": {"values": ["AsinhTransform"]},
        "feature_scaler_map": {
            "values": [
                {
                    "AsinhTransform": [
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
                        "lr_wdi_sp_dyn_imrt_fe_in",
                        "lr_wdi_sh_sta_stnt_zs",
                        "lr_wdi_sh_sta_maln_zs",
                    ],
                    "MinMaxScaler": [
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
        "num_stacks": {"values": [2, 3]},
        # num_blocks: Blocks per stack. N-BEATS paper uses 1 per stack
        # for generic. Keep at 1 — increasing stacks is more effective
        # than increasing blocks, and 2 blocks doubles params per stack.
        "num_blocks": {"values": [2, 4]},
        # num_layers: FC layers per block. 2-4 is standard. Deeper blocks
        # capture more complex patterns but risk overfitting on ~200 series.
        "num_layers": {"values": [2, 3]},
        # layer_widths: Width of FC layers in each block. N-BEATS flattens
        # input_chunk_length * n_features into a single vector (~36×40=1440
        # dims), so layers must be wide enough to avoid crushing that signal.
        # 512-768 keeps compression ratio manageable (~2-3x).
        "layer_widths": {"values": [64, 128, 256]},
        # expansion_coefficient_dim: Dimensionality of basis expansion
        # coefficients (generic mode). Controls expressiveness of the
        # learned basis functions. 5 is paper default, 32 is richer.
        "expansion_coefficient_dim": {"values": [16, 32, 64]},
        # trend_polynomial_degree: Only used in interpretable mode.
        # Included for completeness; irrelevant when generic=True.
        "trend_polynomial_degree": {"values": [2]},
        # activation: ReLU is N-BEATS paper default. LeakyReLU prevents
        # dead neurons on sparse targets.
        "activation": {"values": ["ReLU", "LeakyReLU"]},
        "use_static_covariates": {"values": [True]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        # Dropout: N-BEATS is a deep MLP — moderate dropout needed for
        # ~200 series. Paper uses 0.0 but they had much more data.
        "dropout": {"values": [0.15, 0.25, 0.35]},
        # ==============================================================================
        # LOSS FUNCTION: SpotlightLoss
        # ==============================================================================
        "loss_function": {"values": ["SpotlightLoss"]},
        # ── alpha (truth-only spotlight scale) ───────────────────────────────────────
        # 1+log_cosh(alpha*|y|) — truncated-inverse-density weight (Liu & Lin 2022;
        # Yang et al. 2021 LDS). No pred-side weight — gradient bounded by w(y)×tanh.
        # Weight at max UCDP (asinh≈11.5):
        #   alpha=0.15 → ≈2.1×   alpha=0.25 → ≈3.2×   alpha=0.35 → ≈4.3×
        # GRADIENT BUDGET: alpha scales pointwise gradient magnitude. Capped at 0.35
        # (4.3× max weight) so the pointwise-to-spectral gradient ratio stays in
        # [2:1, 6:1] across the full delta range. alpha=0.5 was 6.1× — starved
        # spectral of gradient budget at low delta, causing it to be ignored.
        # Test run anchor: alpha=0.2, delta=0.15 → balanced.
        "alpha": {
            "distribution": "uniform",
            "min": 0.15,
            "max": 0.35,
        },
        "non_zero_threshold": {"values": [0.88]},  # asinh(1) ≈ 0.88, i.e. ≥1 battle-related death
        # ── delta (multi-resolution spectral weight) ─────────────────────────────────
        # Spectral L1-magnitude matching (n_fft=6,12,24). Phase-insensitive by
        # the Fourier shift theorem: onset 1-mo early → ~zero spectral penalty.
        # n_fft=12 bin 1 = 12-month annual cycle — directly penalises missing seasonality.
        # n_fft=24 catches slow monotonic drift (smooth hockey sticks TV couldn't detect).
        # GRADIENT BUDGET: STFT accumulates ~48 gradient paths per time step across
        # 3 resolutions (8+14+26 bins×frames) vs 1 for pointwise. After .mean()
        # normalisation, spectral gradient norm is ~5-10× pointwise before delta.
        #   delta=0.08 → spectral ≈10-15% of total gradient (light regularisation)
        #   delta=0.15 → spectral ≈20-30% of total gradient (test run anchor)
        #   delta=0.25 → spectral ≈35-45% of total gradient (heavy temporal shaping)
        # Floor at 0.08 so spectral is never noise. Cap at 0.25 so pointwise
        # accuracy isn't starved — the model still needs to get cell values right.
        "delta": {
            "distribution": "uniform",
            "min": 0.08,
            "max": 0.25,
        },
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config
