def get_sweep_config():
    """
    meow
    """
    sweep_config = {
        "method": "bayes",
        "name": "smol_cat_tide_spotlight_v23_cm_msle",
        "early_terminate": {"type": "hyperband", "min_iter": 30, "eta": 2},
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        "input_chunk_length": {"values": [36]},
        "output_chunk_shift": {"values": [0]},
        "random_state": {"values": [67]},
        "output_chunk_length": {"values": [36]},
        "optimizer_cls": {"values": ["Adam"]},
        "mc_dropout": {"values": [False]},
        "num_samples": {"values": [1]},
        "n_jobs": {"values": [-1]},
        # ==============================================================================
        # TRAINING
        # ==============================================================================
        # Batch size: Fixed at 64. MAAT magnitude weighting + hurdle produces
        # heterogeneous per-sample gradients — need enough samples for stable
        # batch gradient estimates. Country-level has fewer series than PGM,
        # so 64 is appropriate.
        "batch_size": {"values": [64]},
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [40]},
        "early_stopping_min_delta": {"values": [0.0001]},
        "force_reset": {"values": [True]},
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        # LR: MAAT has more gradient sources (4 components) — slightly wider
        # range to accommodate different component balance regimes.
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

        # Not relevant. Remove from reproducability gate ----------------------------
        "lr_scheduler_factor": {"values": [0.46]},
        "lr_scheduler_patience": {"values": [7]},
        "lr_scheduler_min_lr": {"values": [1e-5]},
        # ----------------------------
        # MAAT: cosh weight is capped at w_max (default 100), and Huber base
        # limits gradient growth. Lower clip than raw JATLoss needed.
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
        # TiDE ARCHITECTURE
        # ==============================================================================
        # Country-month: fewer series (~200) but richer temporal structure.
        # Need sufficient capacity to model diverse country trajectories.
        "num_encoder_layers": {"values": [1]},
        # Decoder layers: 2 or 3. Country series are smoother than PGM —
        # 2 layers may suffice, but 3 gives more capacity for diverse patterns.
        "num_decoder_layers": {"values": [2, 3]},
        "decoder_output_dim": {"values": [64]},
        # hidden_size: SWEPT. Country-level needs capacity for ~200 diverse
        # trajectories. 256 is minimum viable, 512 gives headroom.
        "hidden_size": {"values": [256, 512]},
        # temporal_width: Country series have stronger annual cycles.
        # 4 (paper default) vs 12 (annual cycle match).
        "temporal_width_past": {"values": [4, 12]},
        "temporal_width_future": {"values": [36, 48]},
        "temporal_decoder_hidden": {"values": [256]},
        "temporal_hidden_size_past": {"values": [128]},
        "temporal_hidden_size_future": {"values": [128]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        "use_layer_norm": {"values": [True]},
        # Dropout: Country-level has fewer training windows per series.
        # Slightly higher dropout ceiling to prevent overfitting on ~200 series.
        "dropout": {
            "distribution": "uniform",
            "min": 0.05,
            "max": 0.20,
        },
        "use_static_covariates": {"values": [True]},
        "use_reversible_instance_norm": {"values": [False]},
        # ==============================================================================
        # LOSS FUNCTION: SpotlightLoss
        # ==============================================================================
        # Four-component loss for asinh-transformed zero-inflated data:
        #   A. Magnitude-recovering weighted Huber (cosh Jacobian weight)
        #   B1. CDF temporal alignment (Cramér distance on cumulative sums)
        #   B2. Temporal derivative penalty (cosine-sim on first-differences)
        #   C. Asymmetric soft-focal classification (hurdle)
        #
        # Stability: w_max caps Jacobian weight. 99.9th-percentile per-sample
        # clamp inside loss. gradient_clip_val ≥ 1.0 externally.
        "loss_function": {"values": ["SpotlightLoss"]},
        # ── alpha (magnitude expansion rate) ──────────
        # Controls cosh amplification. 
        # Country-month data has high max values (asinh~9+).
        #   0.5: cosh(0.5*9) ≈ 45x (Stable, moderate tail pressure)
        #   0.8: cosh(0.8*9) ≈ 222x (Aggressive tail pressure)
        # Fixed < 1.0 to prevent explosion without internal clamping.
        "alpha": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 0.8,
        },
        
        # ── beta (asymmetry strength) ─────────────────
        # Extra multiplier for FN, gated by magnitude.
        #   0.3: FN costs 1.3x FP (on events)
        #   0.7: FN costs 1.7x FP (on events)
        # Range is conservative because magnitude weights already 
        # heavily favor FN recall.
        "beta": {
            "distribution": "uniform",
            "min": 0.3,
            "max": 0.7,
        },
        
        # ── kappa (sigmoid sharpness) ─────────────────
        # Controls transition smoothness between FP/FN regimes.
        #   5.0: Smooth transition.
        #   15.0: Sharp, almost binary transition.
        "kappa": {
            "distribution": "uniform",
            "min": 5.0,
            "max": 15.0,
        },
        
        # ── delta (huber threshold) ───────────────────
        # Transition point for quadratic->linear.
        # Sweeping 0.5 to 1.5 allows finding the optimal robustness.
        "delta": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1.5,
        },
        
        # ── gamma (temporal weight) ───────────────────
        # Weight for the temporal gradient alignment term.
        #   0.05: Light timing guidance.
        #   0.2: Strong timing guidance.
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