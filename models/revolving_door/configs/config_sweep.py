
def get_sweep_config():
    """
    N-HiTS + SpotlightLoss Sweep Configuration
    ============================================

    Data: ~200 country time series, 68 features (conflict + WDI + V-Dem),
    3 targets (lr_ged_sb/ns/os), 86-94% zeros, 36-month horizon.

    N-HiTS is feedforward (no BPTT) — no gradient attenuation across time.

    Architecture (Challu et al. 2022):
    - 2 stacks: structural trends (pooled) + monthly dynamics (raw)
    - AvgPool preferred over MaxPool for zero-inflated data (MaxPool amplifies
      sparse spikes into pooling representation, distorting coarse stack gradients)
    - RevIN ON: empirically required — without it, sinh inversion on OOD predictions
      causes multi-billion-scale outputs. RevIN's sigma-scaling bounds outputs to
      each series' historical variance. Incompatible with SpotlightLoss in principle
      (see notes) but necessary for numerical stability at inference.

    SpotlightLoss parameter constraints (from empirical OOD history):
    - alpha >  0.20 caused OOD explosions even with gradient clipping
    - gamma >= 0.05 is the primary OOD driver via temporal gradient compounding;
      second-order curvature term has been removed from the loss entirely
    - kappa >= 10 creates a near-binary asymmetry gate; combined with beta > 0
      causes systematic overshooting of conflict cells into shared N-HiTS weights
    - delta does NOT exist in SpotlightLoss (was a WeightedPenaltyHuber param)
    """

    sweep_config = {
        "method": "bayes",
        "name": "revolving_door_nhits_spotlight_v6_msle",
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
        # N-HiTS paper uses LR ~1e-3 but SpotlightLoss cosh amplification
        # injects higher gradient variance — keep ceiling conservative.
        "lr": {
            "distribution": "log_uniform_values",
            "min": 5e-5,
            "max": 5e-4,
        },
        "weight_decay": {"values": [5e-6, 5e-3]},
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["CosineAnnealingWarmRestarts"]},
        "lr_scheduler_T_0": {"values": [30]},
        # T_mult=2: progressive lengthening (30→60→120) lets the model
        # settle into finer optima in later cycles.
        "lr_scheduler_T_mult": {"values": [2]},
        "lr_scheduler_eta_min": {"values": [1e-6]},
        # With alpha <= 0.20, max cosh contribution at asinh(10000)=9.9 is
        # cosh(0.20*9.9)≈3.8. Tanh gradient is bounded at ±1. Effective max
        # gradient ≈ 3.8 — clip of 2.0 is sufficient. 5.0 provides no protection.
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
        # N-HiTS ARCHITECTURE
        # ==============================================================================
        # 3 stacks: coarse (semi-annual) + intermediate (quarterly) + fine (monthly).
        # Pooling kernel rationale (icl=36):
        #   kernel=8 → ~quarterly aggregation over the input window
        #   kernel=4 → semi-annual aggregation
        #   kernel=1 → raw monthly (fine stack always)
        # n_freq_downsample rationale (ocl=36):
        #   fd=6 → 36/6 = 6 basis functions  (slow structural trends)
        #   fd=3 → 36/3 = 12 basis functions (quarterly detail)
        #   fd=1 → 36/1 = 36 basis functions (full monthly)
        "num_stacks": {"values": [3]},
        "pooling_kernel_sizes": {"values": [[[8], [4], [1]]]},
        "n_freq_downsample": {"values": [[[6], [3], [1]]]},
        # AvgPool: for zero-inflated data, MaxPool selects the single largest
        # value in each kernel window, making the coarse stack representation
        # dominated by rare spikes. AvgPool produces a smoother structural
        # trend signal that better represents the underlying conflict trajectory.
        "max_pool_1d": {"values": [False, True]},
        "activation": {"values": ["GELU"]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        # Best run hit ceiling (0.143/0.15). N-HiTS's pooling already
        # regularizes, but wider FC layers benefit from more dropout.
        "dropout": {
            "distribution": "uniform",
            "min": 0.10,
            "max": 0.25,
        },
        "use_static_covariates": {"values": [True]},
        # RevIN must be True: empirically, turning it off caused >15B predictions.
        # RevIN's inverse() rescales outputs by each series' own historical sigma,
        # bounding OOD extrapolation relative to observed history.
        "use_reversible_instance_norm": {"values": [True, False]},
        # ==============================================================================
        # LOSS FUNCTION: SpotlightLoss
        # ==============================================================================
        # N-HiTS is feedforward — no BPTT attenuation. Same SpotlightLoss
        # parameter philosophy as smol_cat (TiDE): alpha can be used freely
        # for cosh magnitude weighting.
        "loss_function": {"values": ["SpotlightLoss"]},
        # ── alpha (magnitude expansion rate) ──────────
        # Pre-Basu gate: alpha > 0.20 caused 15B OOD on NHiTS (shared basis
        # contamination). Post-Basu gate: early-training amplification is
        # suppressed (gate≈0.008 at z=5), so higher alpha is now safer to
        # explore without contaminating shared basis functions.
        # The gate fully opens at late training, so alpha still controls final
        # late-training amplification. Cap at 0.50 to limit fully-converged
        # weights: cosh(0.50*9.9)≈74× vs 2900× at alpha=0.80.
        #   0.15: cosh(1.5)≈2.4×  (conservative, was the old safe ceiling)
        #   0.35: cosh(3.5)≈17×   (moderate, smol_cat neighbourhood)
        #   0.50: cosh(5.0)≈74×   (strong, upper safe limit with gate)
        "alpha": {
            "distribution": "uniform",
            "min": 0.10,
            "max": 0.50,
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
        # ModelCatalog builds the encoder dict from this flag at model-build
        # time, selecting functions based on config["level"] — JSON-safe.
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config