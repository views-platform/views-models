def get_sweep_config():
    """BlockRNN"""
    sweep_config = {
        "method": "bayes",
        "name": "dancing_queen_blockrnn_shadow_20260508_D",
        "early_terminate": {"type": "hyperband", "min_iter": 35, "eta": 2},  # min_iter=35: survives 2 RLROP halvings (p=15, cd=3) before bracket cull
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        # 36 for BPTT safety, 48 for full context (Bayes decides).
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
        "batch_size": {"values": [128]},
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [50]},
        # 0.0001 too tight: loss_stability/cv=0.33 → nearly every epoch triggers patience increment
        # even on genuinely improving runs. 0.001 matches all other models in the sweep.
        "early_stopping_min_delta": {"values": [0.001]},
        "force_reset": {"values": [True]},
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        # lr=1e-3 removed: halved 2× by epoch 31 (window 0−10 @ 1e-3, 10−24 @ 5e-4, 24+ @ 2.5e-4)
        # — 2 halvings compresses productive learning into first 24 epochs, Hyperband kills at bracket.
        # 1.5e-4 added: first halving ~epoch 16 → 7.5e-5, longer active learning window before plateau.
        "lr": {"values": [1.5e-4, 2e-4, 5e-4]},
        # weight_decay: 1e-3 crushes FC decoder output weights needed for high-magnitude conflict
        # predictions (weight_max 37→18 observed with 10× more decay). 0 risks overfitting to the
        # 90% zero majority — BPTT gradient signal for zeros outnumbers conflict steps ~9:1.
        "weight_decay": {"values": [2e-4, 1e-4]},
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["ReduceLROnPlateau"]},
        "lr_scheduler_factor": {"values": [0.5]},
        # patience=[15,25]: same dead-config issue as elastic_heart. threshold=0.01
        # in lr_scheduler_kwargs never reaches PyTorch; default 1e-4 fires 100× more
        # aggressively. patience=25 preserves LR budget.
        "lr_scheduler_patience": {"values": [15, 25]},
        "lr_scheduler_min_lr": {"values": [1e-5]},
        "lr_scheduler_kwargs": {"values": [
            {"mode": "min", "factor": 0.5, "patience": 15, "min_lr": 1e-5, "threshold": 0.01, "threshold_mode": "rel", "cooldown": 3},
        ]},
        "gradient_clip_val": {"values": [20, 50]},
        # ==============================================================================
        # SCALING
        # ==============================================================================
        "feature_scaler": {"values": [None]},
        "target_scaler": {"values": ["AsinhTransform"]},  # asinh(x): SpotlightLoss operates in asinh space. non_zero_threshold=0.88=asinh(1).
        "feature_scaler_map": {
            "values": [{
                # Group 1: Zero-Anchor Preservation (Conflict & Heavy Macro)
                # Asinh compresses tails; MaxAbsScaler scales to [-1, 1] based on
                # max absolute value. StandardScaler shifts zero-anchor and creates
                # non-zero mean for peace series — GRU hidden state carries a
                # persistent positive bias from the input mean.
                "AsinhTransform->MaxAbsScaler": [
                    # Conflict counts + deltas + spatial lags
                    "lr_ged_ns", "lr_ged_os",
                    "lr_ged_sb_delta", "lr_ged_ns_delta", "lr_ged_os_delta",
                    "lr_acled_sb", "lr_acled_sb_count", "lr_acled_os",
                    "lr_splag_1_ged_sb", "lr_splag_1_ged_ns", "lr_splag_1_ged_os",

                    # Decay features — conflict regime memory ∈ [0,1]
                    "lr_decay_ged_sb_5", "lr_decay_ged_sb_100", "lr_decay_ged_sb_500",
                    "lr_decay_ged_os_5", "lr_decay_ged_os_100",
                    "lr_decay_ged_ns_5", "lr_decay_ged_ns_100",
                    "lr_decay_acled_sb_5", "lr_decay_acled_os_5", "lr_decay_acled_ns_5",
                    "lr_splag_1_decay_ged_sb_5", "lr_splag_1_decay_ged_os_5", "lr_splag_1_decay_ged_ns_5",

                    # lr_ged temporal lags — explicit trajectory for TiDE (no recurrence)
                    "lr_ged_sb_tlag_1", "lr_ged_sb_tlag_2", "lr_ged_sb_tlag_3",
                    "lr_ged_sb_tlag_4", "lr_ged_sb_tlag_5", "lr_ged_sb_tlag_6",
                    "lr_ged_os_tlag_1",

                    # Topic/NLP features — monthly leading indicators
                    "lr_topic_tokens_t1", "lr_topic_tokens_t2",
                    "lr_topic_ste_theta4_stock_t1", "lr_topic_ste_theta4_stock_t2", "lr_topic_ste_theta4_stock_t13",
                    "lr_topic_ste_theta2_stock_t1", "lr_topic_ste_theta2_stock_t2", "lr_topic_ste_theta2_stock_t13",
                    "lr_topic_ste_theta4_stock_t1_splag", "lr_topic_ste_theta2_stock_t1_splag",

                    # WDI (8 with static covs)
                    "lr_wdi_sm_pop_refg_or", "lr_wdi_sm_pop_netm",
                    "lr_wdi_dt_oda_odat_pc_zs", "lr_wdi_ms_mil_xpnd_gd_zs",
                    "lr_wdi_sp_pop_grow",
                    "lr_wdi_sp_urb_totl_in_zs",
                    "lr_wdi_sp_dyn_imrt_fe_in",
                    "lr_wdi_sh_sta_maln_zs",

                    # V-Dem (12 — pruned of redundant accountability/exclusion)
                    "lr_vdem_v2x_horacc", "lr_vdem_v2x_veracc",
                    "lr_vdem_v2xnp_client", "lr_vdem_v2xnp_regcorr",
                    "lr_vdem_v2xpe_exlgeo", "lr_vdem_v2xpe_exlsocgr",
                    "lr_vdem_v2x_ex_party", "lr_vdem_v2x_ex_military",
                    "lr_vdem_v2xeg_eqdr",
                    "lr_vdem_v2xcl_prpty", "lr_vdem_v2xcl_dmove", "lr_vdem_v2x_clphy",
                ],
            }],
        },
        # ==============================================================================
        # BLOCKRNN ARCHITECTURE
        # ==============================================================================
        # GRU only: LSTM's separate cell+hidden state doubles static-embedding capacity.
        # With 22/31 features annual (constant within the 36-step window), LSTM's forget
        # gate latches the country fingerprint in cell state and ignores weak monthly
        # updates. GRU's single update gate is more permeable to low-amplitude temporal
        # changes from the ~9 monthly features.
        "rnn_type": {"values": ["GRU"]},
        "hidden_dim": {"values": [128, 256]},
        # n_rnn_layers=1 only: second layer receives h1_t which is nearly constant
        # across t (22 static features dominate). Layer 2 learns to deepen the country
        # embedding, not extract temporal patterns — pure leakage capacity.
        "n_rnn_layers": {"values": [1]},
        # FC decoder: h_T is a country embedding when inputs are mostly static.
        # [128] and [128,64] give 40K+ params to learn country→prediction functions.
        # [] (linear readout) and [64] (one small layer) limit decoder complexity
        # without removing the non-linear readout option entirely.
        "hidden_fc_sizes": {"values": [[], [64]]},
        # GELU: smoother gradients through the FC decoder.
        "activation": {"values": ["GELU"]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================

        "dropout": {"values": [0.10, 0.15, 0.25]},
        "use_static_covariates": {"values": [True]},
        "use_reversible_instance_norm": {"values": [True]},
        "loss_function": {"values": ["SpotlightLossLogcosh"]},
        "non_zero_threshold": {"values": [0.88]}, 
        "delta": {"values": [0.0, 0.01]},
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        # use_cyclic_encoders=False: sin/cos(month) covariates dominate h_T after RevIN mean-strip
        # → FC decoder projects a sinusoidal hidden state → 3-period annual sine wave forecasts.
        # Confirmed from HP run: cyclic=True + RevIN = pure sine output regardless of clip/LR.
        "use_cyclic_encoders": {"values": [False]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config