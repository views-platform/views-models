def get_sweep_config():
    """
    Key changes from v3:
    - Wider FC decoder ([256], [128,64]) — v3's [64]/[128] had < 1 hidden unit
      per output, killing target-specific discrimination.
    - Higher beta [0.4, 0.8] — v3 network CAN discriminate (unlike v2's 1.9M
      blob), so stronger asymmetry won't cause blanket over-prediction.
    - Higher delta [2.0, 4.0] — with alpha≈0, delta IS the max gradient from
      any error. v3's delta=1.0 capped Sudan's gradient = Chad's gradient.
      This is why it chronically under-predicts large events.
    - Sweep GRU vs LSTM — GRU's update gate may handle zero-heavy sequences
      better than LSTM's forget gate (which aggressively zeros cell state
      during long runs of zeros).
    - Shorter input option [24, 36] — less zero-dilution of the hidden state.
    - Dropped input_chunk=48 (36 BPTT steps is already the limit for stable
      gradients with SpotlightLoss).
    """
    sweep_config = {
        "method": "bayes",
        "name": "dancing_queen_blockrnn_spotlight_v4_msle",
        "early_terminate": {"type": "hyperband", "min_iter": 30, "eta": 2},
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        # Shorter input reduces zero-dilution of hidden state. 24 months
        # still captures conflict escalation cycles while halving the BPTT
        # path length (→ healthier gradients). 36 for full 3-year context.
        "input_chunk_length": {"values": [24, 36]},
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
            "min": 1e-4,
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
        # Sweep GRU vs LSTM: GRU's single update gate may handle zero-heavy
        # sequences better — LSTM's forget gate learns to aggressively zero
        # the cell state during long runs of zeros, biasing toward under-prediction.
        "rnn_type": {"values": ["GRU"]},
        # hidden_dim: 128-192 for ~200 series. 1-layer keeps all params
        # reachable by gradient.
        "hidden_dim": {"values": [128, 192]},
        "n_rnn_layers": {"values": [1, 2]},
        # hidden_fc_sizes: WIDE decoder. v3's [64]/[128] had less than 1
        # hidden unit per output (108 = 36×3). The decoder must learn
        # target-specific AND step-specific patterns:
        # [256] — single wide layer, 2.4 units per output
        # [128, 64] — two-layer decoder, progressive compression
        "hidden_fc_sizes": {"values": [[256], [128, 64]]},
        # GELU only — smoother gradients through the wider FC decoder.
        # ReLU dead neurons are worse with higher delta/beta pushing
        # gradient magnitudes up.
        "activation": {"values": ["GELU"]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        # Dropout for MC uncertainty via inter-layer RNN dropout.
        # Only active when n_rnn_layers=2 (PyTorch applies dropout between
        # layers only). Bayes will learn to pair n_rnn_layers=2 with
        # dropout > 0 and n_rnn_layers=1 with dropout=0 (no-op anyway).
        "dropout": {"values": [0.05, 0.10]},
        "use_static_covariates": {"values": [True]},
        "use_reversible_instance_norm": {"values": [False]},
        # ==============================================================================
        # LOSS FUNCTION: SpotlightLoss
        # ==============================================================================
        "loss_function": {"values": ["SpotlightLoss"]},
        # ── alpha (magnitude expansion rate) ──────────
        # Near-zero for BPTT safety. 0.0 disables cosh weighting entirely.
        # 0.1 gives cosh(0.1*9) ≈ 1.4x — negligible but lets Bayes test.
        "alpha": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 0.8,
        },
        # ── beta (asymmetry strength) ─────────────────
        "beta": {
            "distribution": "uniform",
            "min": 0.4,
            "max": 0.8,
        },
        # ── kappa (sigmoid sharpness) ─────────────────
        # Moderate. Sharp transitions compound through BPTT.
        "kappa": {
            "distribution": "uniform",
            "min": 5.0,
            "max": 15.0,
        },
        # ── delta (huber threshold) ───────────────────
        "delta": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1.5,
        },
        # ── gamma (temporal weight) ───────────────────
        # Low — block model produces all 36 outputs at once, so temporal
        # smoothness is less natural than for autoregressive models.
        # Just enough to prevent wild step-to-step swings.
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