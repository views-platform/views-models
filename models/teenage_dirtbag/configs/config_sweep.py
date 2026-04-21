def get_sweep_config():
    """
    meow
    """
    sweep_config = {
        "method": "bayes",
        "name": "teenage_dirtbag_tcn_spotlight_v5_msle",
        "early_terminate": {"type": "hyperband", "min_iter": 50, "eta": 2},  # 50 > CAWR T_0=30 — avoids terminating runs at the LR spike before they recover
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        # There are TWO RF constraints that must both hold:
        #   (1) RF ≥ ocl=36 — model must be able to condition all 36 forecast steps
        #       on past observations. Violated by num_layers=4 (RF=31 < 36).
        #   (2) RF ≤ icl    — model should not look past the input window into padding.
        #       Violated by num_layers=5 + icl=48 (RF=63 > 48, but non-fatal).
        # With num_layers=5 fixed: icl=48 wastes ~15 time steps to padding (tolerable);
        # icl=72 satisfies both constraints cleanly (RF=63 ≤ 72, RF=63 ≥ 36).
        "input_chunk_length": {"values": [72]},
        "output_chunk_length": {"values": [36]},
        "output_chunk_shift": {"values": [0]},
        "random_state": {"values": [67]},
        "mc_dropout": {"values": [False]},
        "optimizer_cls": {"values": ["AdamW"]},
        "num_samples": {"values": [1]},
        "n_jobs": {"values": [-1]},

        "time_steps": {"values": [36]},  # Checksum: Must match len(steps)
        "rolling_origin_stride": {"values": [1]},
        "prediction_format": {"values": ["dataframe"]},

        # ==============================================================================
        # TRAINING
        # ==============================================================================
        "batch_size": {"values": [64]},
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [50]},  # CAWR T_0=30: restart spikes LR at epoch 30, patience<50 fires before recovery
        "early_stopping_min_delta": {"values": [0.0001]},
        "force_reset": {"values": [True]},
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        # TCNs are convolution-based — can tolerate slightly higher LR than
        # attention models, but SpotlightLoss multi-component gradients need
        # care. 5e-5 to 1e-3 is log-centered around the ~3e-4 anchor.
        "lr": {
            "distribution": "log_uniform_values",
            "min": 5e-5,
            "max": 1e-3,
        },
        # [0, 1e-5, 1e-4]: 1e-4 is the canonical AdamW value; 0 lets Bayes test no
        # decay; a single value wastes a Bayes parameter slot on a fixed constant.
        "weight_decay": {"values": [1e-4]},
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["CosineAnnealingWarmRestarts"]},
        "lr_scheduler_T_0": {"values": [30]},
        "lr_scheduler_T_mult": {"values": [2]},
        "lr_scheduler_eta_min": {"values": [1e-6]},
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
        # TCN ARCHITECTURE
        # ==============================================================================
        # kernel_size: Small kernels better for sparse signals (Bai et al., 2018).
        # Each kernel sees fewer timesteps, reducing zero-dilution.
        # k=5 excluded: with num_layers=5 gives RF=125 >> icl=72.
        "kernel_size": {"values": [3]},
        # num_filters: 32 filters is a 30:1 compression of the dilated conv stack on
        # sparse country-month data — not enough capacity to fit the rare-event signal.
        # 64-128 is the workable range with weight_norm constraining magnitudes.
        "num_filters": {"values": [128]},
        # dilation_base: Fixed at 2 (standard exponential dilation, Bai et al. 2018).
        # RF formula below assumes d=2 — do not sweep without updating the RF table.
        "dilation_base": {"values": [2]},
        # num_layers: Controls receptive field. TWO constraints must both hold:
        #   RF ≥ ocl=36 (can condition full forecast horizon on past observations)
        #   RF ≤ icl    (no zero-padding waste)
        # RF = 1 + (k-1) × Σ_{i=0}^{L-1}(d^i). With k=3, d=2:
        #   4 layers: RF = 31  months  (RF < ocl=36 ✗ — can't condition full forecast)
        #   5 layers: RF = 63  months  (RF ≥ ocl=36 ✓, RF ≤ icl=72 ✓)
        #   6 layers: RF = 127 months  (RF > icl=72 ✗ — excluded)
        # num_layers=4 was removed: RF=31 < ocl=36 means the model extrapolates blind
        # for the last 5 forecast steps regardless of icl.
        "num_layers": {"values": [5]},
        # weight_norm: Fixed True — constrains filter magnitudes to 1 by construction.
        # Without it, activation spikes from rare non-zero events propagate unboundedly
        # through 5 dilated conv layers, causing the blowups observed in v3 sweep.
        "weight_norm": {"values": [True]},
        # use_reversible_instance_norm: Normalizes input, reverses on output.
        # Helps with distribution shift across countries.
        "use_reversible_instance_norm": {"values": [True]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        # Dropout: TCN applies spatial dropout between conv layers.
        "dropout": {"values": [0.15, 0.25, 0.35]},
        # ==============================================================================
        # LOSS FUNCTION: SpotlightLoss
        # ==============================================================================
        "loss_function": {"values": ["SpotlightLoss"]},
        # ── alpha (truth-only spotlight scale) ───────────────────────────────────────
        # 1+log_cosh(alpha*|y|) — truncated-inverse-density weight (Liu & Lin 2022;
        # Yang et al. 2021 LDS). No pred-side weight — gradient bounded by w(y)×tanh.
        # Weight at max UCDP (asinh≈11.5):
        #   alpha=0.15 → ≈2.1×   alpha=0.25 → ≈3.2×   alpha=0.35 → ≈4.3×
        # GRADIENT BUDGET: alpha scales pointwise gradient magnitude. Capped at 0.35
        # (4.3× max weight) so the pointwise-to-spectral gradient ratio stays in
        # [2:1, 6:1] across the full delta range. alpha=0.5 was 6.1× — starved
        # spectral of gradient budget at low delta, causing it to be ignored.
        # Test run anchor: alpha=0.2, delta=0.15 → balanced.
        "alpha": {
            "distribution": "uniform",
            "min": 0.15,
            "max": 0.35,
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
            "max": 0.25,
        },
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config