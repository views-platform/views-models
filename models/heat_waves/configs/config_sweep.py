def get_sweep_config():
    """
    meow
    """
    sweep_config = {
        "method": "bayes",
        "name": "heat_waves_tft_spotlight_v1_msle",
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
        # TFT combines LSTM + attention — more complex gradient landscape
        # than pure Transformers. Needs conservative LR.
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 3e-4,
        },
        "weight_decay": {"values": [5e-6, 1e-4]},
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["CosineAnnealingWarmRestarts"]},
        "lr_scheduler_T_0": {"values": [30]},
        "lr_scheduler_T_mult": {"values": [2]},
        "lr_scheduler_eta_min": {"values": [1e-6]},
        # TFT has both LSTM and attention gradients — moderate clipping.
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
        # TFT ARCHITECTURE
        # ==============================================================================
        # hidden_size: Controls VSN, GRN, LSTM, and attention dimensions.
        # TFT is parameter-inefficient — hidden_size propagates into every
        # sub-module (LSTM states, attention projections, GRN 2x expansion).
        # 256 doubles the param count and will memorize ~200 series instantly.
        # 64-128 is the viable range; 64/2=32 and 128/4=32 both satisfy
        # the head_dim >= 32 constraint.
        "hidden_size": {"values": [64, 128]},
        # lstm_layers: 1 is standard (Lim et al. 2021). Attention handles
        # long-range, LSTM provides local context. 2 layers doubles LSTM
        # params — too much for ~200 series.
        "lstm_layers": {"values": [1]},
        # num_attention_heads: 2 for hidden=64 (head_dim=32), 4 for
        # hidden=128 (head_dim=32). Both satisfy >= 32 constraint.
        # Sweeping [2, 4] — Bayesian search pairs with hidden_size.
        "num_attention_heads": {"values": [2, 4]},
        # full_attention: True = O(n²) over full encoder+decoder context.
        # With 36-48 timesteps this is cheap and gives richer attention.
        "full_attention": {"values": [True, False]},
        # feed_forward: GatedResidualNetwork is TFT's native architecture.
        # SwiGLU is a modern alternative worth testing.
        "feed_forward": {"values": ["GatedResidualNetwork"]},
        # hidden_continuous_size: Dimensionality of continuous variable
        # processing before VSN. Should be <= hidden_size. With hidden_size
        # capped at 128, 32-64 is the right range.
        "hidden_continuous_size": {"values": [32, 64]},
        # categorical_embedding_sizes: empty dict for pure continuous features.
        "categorical_embedding_sizes": {"values": [{}]},
        # add_relative_index: Injects position information into attention.
        "add_relative_index": {"values": [True]},
        # skip_interpolation: Skip the interpolation in decoder output.
        "skip_interpolation": {"values": [False]},
        # norm_type: LayerNorm is standard and most stable for TFT.
        "norm_type": {"values": ["LayerNorm"]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        # Dropout: TFT is the most parameter-heavy model in the catalog
        # (VSN + GRN + LSTM + attention). Needs strong regularization
        # with only ~200 series. Floor at 0.15 like Transformer.
        "dropout": {
            "distribution": "uniform",
            "min": 0.15,
            "max": 0.35,
        },
        "use_static_covariates": {"values": [True]},
        "use_reversible_instance_norm": {"values": [False]},
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