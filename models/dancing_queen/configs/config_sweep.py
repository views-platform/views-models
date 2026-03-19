def get_sweep_config():
    """
    meow
    """
    sweep_config = {
        "method": "bayes",
        "name": "dancing_queen_blockrnn_spotlight_v3_msle",
        "early_terminate": {"type": "hyperband", "min_iter": 30, "eta": 2},
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }
    # ==============================================================================
    # WHY alpha MUST BE NEAR-ZERO FOR BlockRNN
    # ==============================================================================
    # BlockRNN processes input through 36 LSTM timesteps, then maps the final
    # hidden state through FC layers to produce all 36 output values. Gradients
    # from SpotlightLoss must travel back through all 36 BPTT steps.
    #
    # At alpha=0.6, a conflict event at m=7 gets cosh(4.2) ≈ 33x weight. After
    # 36 BPTT steps, that gradient is attenuated by the vanishing gradient
    # factor — it might arrive at the input weights as 3x or less. But at
    # alpha=0.8 and m=9, cosh(7.2) ≈ 672x, which after BPTT might still be 50x+.
    #
    # Batches containing a high-magnitude conflict event produce 100x+ larger
    # gradient norms than zero-dominated batches. The optimizer can't form stable
    # parameter updates because the gradient direction and magnitude are dominated
    # by random batch composition. The model's defensive response: collapse to
    # predicting near-zero, which gives consistent low loss on 86% of observations.
    # ==============================================================================
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
        # BLOCKRNN ARCHITECTURE
        # ==============================================================================
        # LSTM: Separate cell state for long-term memory — critical for
        # 36-month horizon where conflict escalation patterns span years.
        "rnn_type": {"values": ["LSTM"]},
        # hidden_dim: RNN hidden state size. 128-192 for ~200 country series.
        # v2 best run (384×2 layers) had median gradient 0.002 — 50% of params
        # received zero signal. Smaller network = every param gets gradient.
        "hidden_dim": {"values": [128, 192]},
        # n_rnn_layers: 1 layer only. 2-layer 384-dim had 1.9M params for
        # ~200 series — gradient starvation in lower layers. 1 layer at
        # 128-192 gives 200-400K params where the entire network trains.
        "n_rnn_layers": {"values": [1]},
        # hidden_fc_sizes: FC decoder after RNN output. REQUIRED — without it
        # the 128/192-dim hidden state projects directly to 108 outputs (36×3)
        # via a single linear layer with no nonlinear bottleneck for
        # target-specific discrimination.
        "hidden_fc_sizes": {"values": [[64], [128]]},
        # activation: GELU provides smoother gradients than ReLU through
        # the FC decoder, reducing dead neuron risk on sparse targets.
        "activation": {"values": ["ReLU", "GELU"]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        # Dropout: RNNs are more prone to overfitting on small series counts.
        # MC dropout enabled — used for uncertainty at inference.
        "dropout": {
            "values": [0.0],
        },
        "use_static_covariates": {"values": [True]},
        # RevIN: Handles distribution shift between train/test periods.
        "use_reversible_instance_norm": {"values": [False]},
        # ==============================================================================
        # LOSS FUNCTION: SpotlightLoss
        # ==============================================================================
        "loss_function": {"values": ["SpotlightLoss"]},
        # ── alpha (magnitude expansion rate) ──────────
        # Must be near-zero for RNN. Cosh creates O(100x) gradient
        # variance that BPTT attenuates unpredictably → mode collapse.
        # 0.0 = disabled, 0.1 = very mild (cosh(0.1*9) ≈ 1.4x — negligible)
        "alpha": {"values": [0.0, 0.1]},
        # ── beta (asymmetry strength) ─────────────────
        # Primary anti-collapse mechanism for RNN.
        # Under-predicting real events costs (1+beta)x more than over-predicting.
        # v2 best run had beta=0.63 → over-predicted on 86% zero-observations
        # (y_hat_bar_sb=22.78 but Pearson=0.21 — no discrimination).
        # Lower range: enough to prevent collapse, not enough to flood zeros.
        "beta": {
            "distribution": "uniform",
            "min": 0.2,
            "max": 0.5,
        },
        # ── kappa (sigmoid sharpness) ─────────────────
        # Keep moderate — sharp transitions (>10) create gradient spikes
        # that compound through BPTT.
        "kappa": {"values": [8.0, 10.0]},
        # ── delta (huber threshold) ───────────────────
        # Wider quadratic region = more stable MSE-like gradients.
        # RNNs need gradient consistency across the batch.
        "delta": {
            "distribution": "uniform",
            "min": 1.0,
            "max": 2.5,
        },

        # ── gamma (temporal weight) ───────────────────
        # Temporal gradient alignment is valuable for RNNs — it provides
        # a supervision signal at every timestep, combating vanishing gradients.
        "gamma": {
            "distribution": "uniform",
            "min": 0.05,
            "max": 0.15,
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