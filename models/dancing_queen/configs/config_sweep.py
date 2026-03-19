def get_sweep_config():
    """
    meow
    """
    sweep_config = {
        "method": "bayes",
        "name": "dancing_queen_blockrnn_spotlight_v1_msle",
        "early_terminate": {"type": "hyperband", "min_iter": 30, "eta": 2},
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        "input_chunk_length": {"values": [36, 48]},
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
        "lr": {
            "distribution": "log_uniform_values",
            "min": 3e-5,
            "max": 3e-4,
        },
        "weight_decay": {"values": [5e-6]},
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["CosineAnnealingWarmRestarts"]},
        "lr_scheduler_T_0": {"values": [30]},
        "lr_scheduler_T_mult": {"values": [2]},
        "lr_scheduler_eta_min": {"values": [1e-6]},
        "gradient_clip_val": {"values": [2.0, 3.0, 5.0]},
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
        # BLOCKRNN ARCHITECTURE
        # ==============================================================================
        # LSTM: Separate cell state for long-term memory — critical for
        # 36-month horizon where conflict escalation patterns span years.
        "rnn_type": {"values": ["LSTM"]},
        # hidden_dim: RNN hidden state size. 128-256 for ~200 country series.
        # Larger than TiDE hidden_size because the RNN must compress the
        # entire temporal context into a single hidden vector.
        "hidden_dim": {"values": [128, 256, 384]},
        # n_rnn_layers: 2 layers for hierarchical pattern extraction.
        # 1 is too shallow for 36-month context, 3+ risks vanishing gradients.
        "n_rnn_layers": {"values": [1, 2]},
        # hidden_fc_sizes: FC layers after RNN output. None uses a single
        # linear projection. [128] adds a hidden layer for richer decoding.
        "hidden_fc_sizes": {"values": [None, [128]]},
        # activation: GELU provides smoother gradients than ReLU through
        # the FC decoder, reducing dead neuron risk on sparse targets.
        "activation": {"values": ["ReLU", "GELU"]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        # Dropout: RNNs are more prone to overfitting on small series counts.
        # MC dropout enabled — used for uncertainty at inference.
        "dropout": {
            "distribution": "uniform",
            "min": 0.05,
            "max": 0.20,
        },
        "use_static_covariates": {"values": [True]},
        # RevIN: Handles distribution shift between train/test periods.
        "use_reversible_instance_norm": {"values": [False]},
        # ==============================================================================
        # LOSS FUNCTION: SpotlightLoss
        # ==============================================================================
        "loss_function": {"values": ["SpotlightLoss"]},
        # ── alpha (magnitude expansion rate) ──────────
        # Controls cosh amplification for asinh-transformed targets.
        #   0.5: cosh(0.5*9) ≈ 45x  (stable, moderate tail pressure)
        #   0.8: cosh(0.8*9) ≈ 222x (aggressive tail pressure)
        "alpha": {
            "distribution": "uniform",
            "min": 0.4,
            "max": 0.8,
        },
        # ── beta (asymmetry strength) ─────────────────
        # Extra FN multiplier gated by magnitude.
        #   0.3: FN costs 1.3x FP on events
        #   0.7: FN costs 1.7x FP on events
        "beta": {
            "distribution": "uniform",
            "min": 0.3,
            "max": 0.7,
        },
        # ── kappa (sigmoid sharpness) ─────────────────
        # Transition smoothness between FP/FN regimes.
        #   5: smooth, 15: near-binary.
        "kappa": {
            "distribution": "uniform",
            "min": 5.0,
            "max": 15.0,
        },
        # ── delta (huber threshold) ───────────────────
        # Quadratic→linear transition point.
        "delta": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1.5,
        },
        # ── gamma (temporal weight) ───────────────────
        # Weight for temporal gradient alignment term.
        "gamma": {
            "distribution": "uniform",
            "min": 0.05,
            "max": 0.2,
        },
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        "add_encoders": {
            "values": [
                {
                    "position": {"past": ["relative"], "future": ["relative"]},
                }
            ]
        },
    }

    sweep_config["parameters"] = parameters
    return sweep_config