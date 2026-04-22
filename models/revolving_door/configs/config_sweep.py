
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
        "name": "revolving_door_nhits_spotlight_v9_msle",
        "early_terminate": {"type": "hyperband", "min_iter": 50, "eta": 2},  # 50 > CAWR T_0=30 — avoids terminating runs at the LR spike before they recover
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        "input_chunk_length": {"values": [48]},
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
        "early_stopping_patience": {"values": [50]},
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
            "max": 1e-3,
        },
        "weight_decay": {"values": [1e-4]},
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
        "gradient_clip_val": {"values": [2.0]},
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
                        "lr_ged_ns", 
                        "lr_ged_os",
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
        # 3 stacks: coarse (annual) + intermediate (quarterly) + fine (monthly).
        # Pooling kernel rationale (icl=48):
        #   kernel=12 → 48/12 = 4 annual groups  (coarse multi-year trends)
        #   kernel=4  → 48/4  = 12 quarterly groups (conflict-cycle patterns)
        #   kernel=1  → 48 raw monthly steps       (fine stack always unpooled)
        # n_freq_downsample rationale (ocl=36, independent of icl):
        #   fd=6 → 36/6 = 6 basis functions  (slow structural trends, semi-annual)
        #   fd=3 → 36/3 = 12 basis functions (quarterly detail)
        #   fd=1 → 36/1 = 36 basis functions (full monthly)
        "num_stacks": {"values": [3]},
        "pooling_kernel_sizes": {"values": [[[12], [4], [1]]]},
        # n_freq_downsample: basis functions per stack = ocl / fd.
        # Option A [[6],[3],[1]]: coarse=6 basis (semi-annual), intermediate=12, fine=36.
        #   Risk: coarse stack has 4 pooled inputs → 6 outputs = underdetermined FC,
        #   enabling spurious high-freq content that compounds across stacks → OOD.
        # Option B [[12],[3],[1]]: coarse=3 basis (annual), intermediate=12, fine=36.
        #   Safer: 4 inputs → 3 outputs = overdetermined FC, structurally constrains
        #   coarse stack to annual-only resolution. Directly addresses y_hat inflation.
        "n_freq_downsample": {"values": [[[12], [3], [1]]]},
        # AvgPool: for zero-inflated data, MaxPool selects the single largest
        # value in each kernel window, making the coarse stack representation
        # dominated by rare spikes. AvgPool produces a smoother structural
        # trend signal that better represents the underlying conflict trajectory.
        "max_pool_1d": {"values": [False]},
        "activation": {"values": ["GELU"]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        "dropout": {"values": [0.15, 0.25, 0.35]},
        "use_static_covariates": {"values": [True]},
        # RevIN must be True: empirically, turning it off caused >15B predictions.
        # RevIN's inverse() rescales outputs by each series' own historical sigma,
        # bounding OOD extrapolation relative to observed history.
        "use_reversible_instance_norm": {"values": [True]},
        # ==============================================================================
        # LOSS FUNCTION: SpotlightLoss
        # ==============================================================================
        # N-HiTS is feedforward — no BPTT attenuation. Same SpotlightLoss
        # parameter philosophy as smol_cat (TiDE): alpha can be used freely
        # for cosh magnitude weighting.
        "loss_function": {"values": ["SpotlightLoss"]},
        "alpha": {
            "distribution": "uniform",
            "min": 0.10,
            "max": 0.30,
        },
        "non_zero_threshold": {"values": [0.88]},  # asinh(1) ≈ 0.88, i.e. ≥1 battle-related death
        # ── delta (multi-resolution spectral weight) ─────────────────────────────────
        # Spectral L1-magnitude matching (n_fft=6,12,24). Phase-insensitive by
        # the Fourier shift theorem: onset 1-mo early → ~zero spectral penalty.
        # n_fft=12 bin 1 = 12-month annual cycle — directly penalises missing seasonality.
        # n_fft=24 catches slow monotonic drift (smooth hockey sticks TV couldn't detect).
        # GRADIENT BUDGET: STFT accumulates ~48 gradient paths per time step across
        # 3 resolutions (8+14+26 bins×frames) vs 1 for pointwise. After .mean()
        # normalisation, spectral gradient norm is ~5-10× pointwise before delta.
        #   delta=0.08 → spectral ≈10-15% of total gradient (light regularisation)
        #   delta=0.15 → spectral ≈20-30% of total gradient (test run anchor)
        #   delta=0.25 → spectral ≈35-45% of total gradient (heavy temporal shaping)
        # Floor at 0.08 so spectral is never noise. Cap at 0.25 so pointwise
        # accuracy isn't starved — the model still needs to get cell values right.
        "delta": {
            "distribution": "uniform",
            "min": 0.08,
            "max": 0.28,
        },
        # ModelCatalog builds the encoder dict from this flag at model-build
        # time, selecting functions based on config["level"] — JSON-safe.
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config