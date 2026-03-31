def get_sweep_config():
    """
    meow
    """
    sweep_config = {
        "method": "bayes",
        "name": "teenage_dirtbag_tcn_spotlight_v2_msle",
        "early_terminate": {"type": "hyperband", "min_iter": 30, "eta": 2},
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        # TCN requires input_chunk_length > output_chunk_length (hard constraint
        # from dilated causal convolutions). icl=72 gives 36 months of lookback
        # beyond the forecast horizon and fits RF ≤ 63 for all valid (k, layers)
        # combinations in this sweep.
        "input_chunk_length": {"values": [72]},
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
        # TCNs are convolution-based — can tolerate slightly higher LR than
        # attention models, but SpotlightLoss multi-component gradients need
        # care. 3e-5 to 5e-4 covers safe range.
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
        # TCN ARCHITECTURE
        # ==============================================================================
        # kernel_size: Small kernels better for sparse signals (Bai et al., 2018).
        # Each kernel sees fewer timesteps, reducing zero-dilution.
        # k=5 excluded: with num_layers=[4,5] gives RF=125/253 >> icl=72.
        "kernel_size": {"values": [3]},
        # num_filters: Number of convolutional filters per layer. 64-128 is
        # moderate capacity for ~200 series — avoids overfitting to dominant
        # zeros while giving enough representational power.
        "num_filters": {"values": [32, 64, 128]},
        # dilation_base: Standard exponential dilation (powers of 2).
        "dilation_base": {"values": [2]},
        # num_layers: Controls receptive field. Constraint is RF ≤ icl (not RF ≥ ocl).
        # RF = 1 + (k-1) × Σ_{i=0}^{L-1}(d^i). With k=3, d=2, icl=72:
        #   4 layers: RF = 1 + 2×(1+2+4+8)      = 31  months  (RF < icl ✓)
        #   5 layers: RF = 1 + 2×(1+2+4+8+16)   = 63  months  (RF < icl ✓)
        #   6 layers: RF = 1 + 2×(1+2+4+8+16+32)= 127 months  (RF > icl ✗ — excluded)
        "num_layers": {"values": [4, 5]},
        # weight_norm: Essential for TCN stability — better than batch_norm
        # for sparse gradients in zero-inflated data.
        "weight_norm": {"values": [True, False]},
        # use_reversible_instance_norm: Normalizes input, reverses on output.
        # Helps with distribution shift across countries.
        "use_reversible_instance_norm": {"values": [True, False]},
        "use_static_covariates": {"values": [True]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        # Dropout: TCN applies spatial dropout between conv layers. With
        # weight_norm already regularizing, keep dropout low—too much kills
        # the rare-event neurons. 0.05-0.15 preserves conflict signal.
        "dropout": {
            "distribution": "uniform",
            "min": 0.05,
            "max": 0.15,
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