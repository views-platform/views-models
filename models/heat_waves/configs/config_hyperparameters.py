def get_hp_config():
    """
    TFT hyperparameters from SpotlightLossLogcosh sweep best run.
    """
    
    hyperparameters = {
        # Temporal
        "steps": [*range(1, 36 + 1, 1)],
        "time_steps": 36,  # Checksum: Must match len(steps)
        "input_chunk_length": 36,
        "output_chunk_length": 36,
        "output_chunk_shift": 0,
        "random_state": 67,

        # Inference
        "num_samples": 1,
        "mc_dropout": False,
        "n_jobs": -1,

        # Training
        "batch_size": 128,
        "n_epochs": 300,
        "early_stopping_patience": 30,
        "early_stopping_min_delta": 0.001,
        "force_reset": True,

        # Optimizer
        "optimizer_cls": "AdamW",
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "gradient_clip_val": 15,
        "optimizer_kwargs": {
            "lr": 5e-4,
            "weight_decay": 1e-4,
        },

        # LR Scheduler
        "lr_scheduler_cls": "ReduceLROnPlateau",
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 10,
        "lr_scheduler_min_lr": 1e-6,
        "lr_scheduler_kwargs": {
            "mode": "min",
            "factor": 0.5,
            "patience": 10,
            "min_lr": 1e-6,
            "cooldown": 3,
            "threshold": 0.01,
            "threshold_mode": "rel",
        },

        # Loss
        "loss_function": "SpotlightLossLogcosh",
        "delta": 0.05544820600171585,
        "non_zero_threshold": 0.88,

        # Scaling
        "feature_scaler": None,
        "target_scaler": "AsinhTransform",
        "feature_scaler_map": {
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
        },

        # TFT Architecture
        # hidden_size=128: VSN flattened_grn receives concat of all 37 variable
        # prescaler embeddings (37 × hidden_continuous_size = 37 × 32 = 1184 dims).
        # At hidden=64, the GRN hidden layer is min(64, 37)=37 → bottleneck collapses
        # the 5 monthly conflict features and 32 annual features into the same 37-dim
        # space with no separation capacity. 128 gives the softmax enough headroom to
        # assign distinct weights to monthly vs annual features.
        "hidden_size": 128,
        # lstm_layers=1: TFT initialises LSTM h_0 and c_0 from static_context_grn
        # (country fingerprint). Layer 2 receives h_1 as input — but h_1 is dominated
        # by the 32 annually-constant features (stable gradients every step). Layer 2
        # learns a deeper country embedding, not temporal dynamics. Combined with the
        # static enrichment GRN (which re-injects country context after LSTM), a 2-layer
        # LSTM double-encodes the country profile. For zero-inflated data where the 5
        # monthly features are the only dynamic signal, single-layer LSTM preserves more
        # bandwidth for those features.
        "lstm_layers": 1,
        # num_attention_heads=2: TFT uses InterpretableMultiHeadAttention — all heads
        # share the same V projection (v_layer), only Q/K are per-head. With 4 heads,
        # d_k=32 (128/4). The Q×K^T dot product in 32-dim space has SNR ∝ 1/√32 ≈ 0.18.
        # For zero-inflated data the conflict timesteps are rare positive values in a
        # near-zero background — a 32-dim attention needs very high conflict amplitude
        # to produce a sharp attention peak above the zero background noise. With 2
        # heads, d_k=64 → SNR ∝ 1/√64 ≈ 0.125 better separation. Final output is
        # mean of 2 heads (less averaging than 4), preserving spike amplitude.
        "num_attention_heads": 2,
        "full_attention": False,
        "feed_forward": "GatedResidualNetwork",
        # hidden_continuous_size=32: each feature is projected 1→32 before VSN.
        # For zero-inflated conflict features (e.g. lr_ged_sb_delta), the prescaler
        # must encode both magnitude and sign in 32 dims. At 16, the per-variable GRN
        # (16→min(16,128)→128) compresses before it can differentiate conflict onset
        # from noise. 32 is h/4 at hidden=128 — standard TFT ratio.
        "hidden_continuous_size": 32,
        # dropout=0.25: TFT applies dropout at 4 independent sites — LSTM recurrent
        # path, GRN gates, VSN softmax inputs, and attention weights. Compound survival
        # rate = (1-d)^4. At d=0.30: 24% survival. At d=0.25: 32% survival.
        # Conflict signal is sparse — an event series has ~30% non-zero timesteps.
        # At d=0.30, a conflict-onset timestep survives all 4 dropout gates with only
        # 24% probability per forward pass. 0.25 improves this to 32% without
        # sacrificing regularisation (the 90% peaceful series still receive heavy
        # regularisation via zero-activation dropout paths).
        "dropout": 0.25,
        "norm_type": "LayerNorm",
        # add_relative_index=True: without position encoding, the attention mask
        # (causal, full_attention=False) prevents future leakage but attention scores
        # are purely content-based. Step 8 and step 20 produce identical Q vectors if
        # the decoder inputs are identical. The model cannot differentiate an early-
        # horizon spike from a late-horizon plateau. Relative index breaks this
        # degeneracy, allowing the attention to resolve temporal position of conflict
        # onset within the 36-step forecast window.
        "add_relative_index": True,
        # skip_interpolation=True: skips the TimeDistributedInterpolation in the
        # ResampleNorm path. Interpolation uses F.interpolate(mode='linear') which
        # applies linear blending across the sequence dimension — directly smoothing
        # conflict spikes in the residual path. Skip=True routes the residual through
        # the trainable gate only (mask parameter × sigmoid × 2.0), preserving
        # spike amplitude in the skip connections.
        "skip_interpolation": True,
        "categorical_embedding_sizes": {},
        "use_static_covariates": True,
        "use_reversible_instance_norm": True,
        "likelihood": None,

        # "static_covariate_stats": {
        #     "transform": "AsinhTransform->MaxAbsScaler",
        #     "inject": False,
        # },
        "checkpoint_mode": "best",

        # Encoders
        "use_cyclic_encoders": False,
    }
    return hyperparameters
