def get_sweep_config():
    """
    meow
    """
    sweep_config = {
        "method": "bayes",
        "name": "new_rules_nbeats_spotlight_v1_msle",
        "early_terminate": {"type": "hyperband", "min_iter": 30, "eta": 2},
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
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
        "batch_size": {"values": [64]},
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [40]},
        "early_stopping_min_delta": {"values": [0.0001]},
        "force_reset": {"values": [True]},
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        # N-BEATS is a pure MLP architecture — simpler gradient landscape.
        # Can tolerate slightly higher LR than attention-based models.
        "lr": {
            "distribution": "log_uniform_values",
            "min": 3e-5,
            "max": 5e-4,
        },
        "weight_decay": {"values": [5e-6]},
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["CosineAnnealingWarmRestarts"]},
        "lr_scheduler_T_0": {"values": [30]},
        "lr_scheduler_T_mult": {"values": [2]},
        "lr_scheduler_eta_min": {"values": [1e-6]},
        "gradient_clip_val": {"values": [1.0, 2.0, 3.0]},
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
        "generic_architecture": {"values": [True, False]},
        # num_stacks: Number of stacks. Each stack processes the residual
        # from the previous. 2 is standard for generic, more adds capacity.
        "num_stacks": {"values": [2, 3]},
        # num_blocks: Blocks per stack. N-BEATS paper uses 1 per stack
        # for generic. Keep at 1 — increasing stacks is more effective
        # than increasing blocks, and 2 blocks doubles params per stack.
        "num_blocks": {"values": [1]},
        # num_layers: FC layers per block. 2-4 is standard. Deeper blocks
        # capture more complex patterns but risk overfitting on ~200 series.
        "num_layers": {"values": [2, 3]},
        # layer_widths: Width of FC layers in each block. N-BEATS flattens
        # input_chunk_length * n_features into a single vector (~36×40=1440
        # dims), so layers must be wide enough to avoid crushing that signal.
        # 512-768 keeps compression ratio manageable (~2-3x).
        "layer_widths": {"values": [256, 512, 768]},
        # expansion_coefficient_dim: Dimensionality of basis expansion
        # coefficients (generic mode). Controls expressiveness of the
        # learned basis functions. 5 is paper default, 32 is richer.
        "expansion_coefficient_dim": {"values": [16, 32]},
        # trend_polynomial_degree: Only used in interpretable mode.
        # Included for completeness; irrelevant when generic=True.
        "trend_polynomial_degree": {"values": [2]},
        # activation: ReLU is N-BEATS paper default. LeakyReLU prevents
        # dead neurons on sparse targets.
        "activation": {"values": ["ReLU", "LeakyReLU"]},
        "use_static_covariates": {"values": [True, False]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        # Dropout: N-BEATS is a deep MLP — moderate dropout needed for
        # ~200 series. Paper uses 0.0 but they had much more data.
        "dropout": {
            "distribution": "uniform",
            "min": 0.05,
            "max": 0.20,
        },
        # ==============================================================================
        # LOSS FUNCTION: SpotlightLoss
        # ==============================================================================
        "loss_function": {"values": ["SpotlightLoss"]},
        # ── alpha (magnitude expansion rate) ──────────
        "alpha": {
            "distribution": "uniform",
            "min": 0.10,
            "max": 0.80,
        },
        
        # ── beta (asymmetry strength) ─────────────────
        # Extra multiplier for FN, gated by magnitude.
        #   0.3: FN costs 1.3x FP (on events)
        #   0.7: FN costs 1.7x FP (on events)
        # Range is conservative because magnitude weights already 
        # heavily favor FN recall.
        "beta": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 0.3,
        },
        
        # ── kappa (sigmoid sharpness) ─────────────────
        # Controls transition smoothness between FP/FN regimes.
        #   5.0: Smooth transition.
        #   15.0: Sharp, almost binary transition.
        "kappa": {
            "distribution": "uniform",
            "min": 8.0,
            "max": 15.0,
        },
        # ── gamma (temporal weight) ───────────────────
        # Weight for the temporal gradient alignment term.
        #   0.05: Light timing guidance.
        #   0.2: Strong timing guidance.
        "gamma": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 0.2,
        },
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config
