def get_sweep_config():
    """
    meow
    """
    sweep_config = {
        "method": "bayes",
        "name": "teenage_dirtbag_tcn_spotlight_20260503",
        "early_terminate": {"type": "hyperband", "min_iter":25, "eta": 2},  # >T_0=25 — avoids terminating runs at the LR spike before they recover
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        # RF constraints with num_layers=5, dilation_base=2:
        #   k=3: RF=63  — exceeds icl=48 by 15 steps (zero-padded peace, tolerable)
        #   k=5: RF=125 — exceeds icl=48 by 77 steps (more padding, still tolerable;
        #        see kernel_size comment for the lag-12 gain that justifies this)
        # icl=48 was chosen over 72: 24 additional mostly-zero steps dilute the
        # conflict signal in the encoder input more than they extend seasonal context.
        "input_chunk_length": {"values": [48]},
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
        "batch_size": {"values": [128]},
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [30]},
        "early_stopping_min_delta": {"values": [1e-3]},
        "force_reset": {"values": [True]},
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        # lr: swept to allow small-filter runs to survive Hyperband's min_iter=25 cut.
        # AdamW bias correction + mostly-zero batches → effective LR starts near 0;
        # 5e-4 is too conservative for 32-filter runs to show signal at epoch 25.
        # 1e-3 allows fast early convergence for small capacity; gradient ceiling
        # is handled by gradient_clip_val — see note there.
        "lr": {"values": [1e-3, 5e-4]},
        # weight_decay: 1e-3 removed — with weight_norm, WD acts on filter magnitude g.
        # ReduceLROnPlateau decays LR (e.g. to 1e-5) but WD=1e-3 continues pulling g→0,
        # silencing filters that learned conflict representations late in training.
        "weight_decay": {"values": [1e-4]},
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["ReduceLROnPlateau"]},
        "lr_scheduler_factor": {"values": [0.7]},
        # patience raised 10→15: with 90% zero inflation, ReduceLROnPlateau fires on
        # the false peace-series plateau. At lr=5e-4 with patience=10, LR had already
        # decayed to 0.000245 by epoch 48 — conflict learning throttled before convergence
        # (train_loss still 2.97 at that point).
        "lr_scheduler_patience": {"values": [15]},
        "lr_scheduler_min_lr": {"values": [1e-6]},
        "lr_scheduler_kwargs": {"values": [{"mode": "min", "factor": 0.7, "patience": 15, "min_lr": 1e-6, "threshold": 0.01, "threshold_mode": "rel", "cooldown": 3}]},
        # gradient_clip_val: Fixed at 1.0. SpotlightLossLogcosh gradient = tanh(e),
        # bounded at ±1 per element by construction. Any grad_norm spike above 1.0 is
        # amplification through the 5-layer conv stack (weight_norm constrains this),
        # not a meaningful error signal from the loss
        "gradient_clip_val": {"values": [1.0]},
        # ==============================================================================
        # SCALING
        # ==============================================================================
        "feature_scaler": {"values": [None]},
        "target_scaler": {"values": ["AsinhTransform"]},  # log1p(x): log-compresses targets, expm1 inverse
        "feature_scaler_map": {
            "values": [{
                "AsinhTransform": [
                    # Heavy-tailed: conflict counts, GDP, refugees, ODA
                    "lr_splag_1_ged_sb", "lr_splag_1_ged_ns", "lr_splag_1_ged_os",
                    "lr_ged_ns", "lr_ged_os",
                    "lr_ged_sb_delta", "lr_ged_ns_delta", "lr_ged_os_delta",
                    "lr_wdi_ny_gdp_mktp_kd", "lr_wdi_nv_agr_totl_kn",
                    "lr_wdi_sm_pop_refg_or", "lr_wdi_dt_oda_odat_pc_zs",
                    "lr_wdi_sp_pop_grow", "lr_wdi_sp_urb_totl_in_zs",
                    "lr_wdi_sm_pop_netm", "lr_acled_sb", 
                    "lr_acled_sb_count", "lr_acled_os",

                    # Bounded [0,1] or near-bounded: V-Dem indices, WDI rates
                    "lr_vdem_v2x_horacc", "lr_vdem_v2x_veracc", "lr_vdem_v2x_diagacc",
                    "lr_vdem_v2xnp_client", "lr_vdem_v2xnp_regcorr",
                    "lr_vdem_v2xpe_exlpol", "lr_vdem_v2xpe_exlgeo",
                    "lr_vdem_v2xpe_exlgender", "lr_vdem_v2xpe_exlsocgr",
                    "lr_vdem_v2x_divparctrl", "lr_vdem_v2x_ex_party",
                    "lr_vdem_v2x_ex_military", "lr_vdem_v2x_genpp",
                    "lr_vdem_v2xeg_eqdr", "lr_vdem_v2xcl_prpty",
                    "lr_vdem_v2xeg_eqprotec", "lr_vdem_v2xcl_dmove",
                    "lr_vdem_v2x_clphy",
                    "lr_wdi_ms_mil_xpnd_gd_zs", "lr_wdi_sh_sta_stnt_zs",
                    "lr_wdi_sh_sta_maln_zs", "lr_wdi_sl_tlf_totl_fe_zs",
                    "lr_wdi_se_enr_prim_fm_zs", "lr_wdi_sp_dyn_imrt_fe_in",
                ],
            }]
        },
        # ==============================================================================
        # TCN ARCHITECTURE
        # ==============================================================================
        # kernel_size: Swept [3, 5]. k=3 (RF=63) is the sparse-signal standard
        # (Bai et al., 2018). k=5 (RF=125) added for annual seasonality: with d=2,
        # layer 2 (dilation=4) directly samples t, t-4, t-8, t-12, t-16 — a
        # first-class lag-12 connection. k=3 has no single layer touching lag-12;
        # annual patterns must emerge from multi-hop combinations, which is structurally
        # harder to learn. RF=125 > icl=48 by 77 steps — zero-padding in the causal
        # window contributes peace baseline (appropriate for 90% zero-inflated data).
        "kernel_size": {"values": [3, 5]},
        # num_filters: 128 removed. Two confirmed logcosh-era blowups: event_ratio=
        # 14,226× (lr=5e-4, dropout=0.35, epoch 88) and 18,342× (lr=1e-3, dropout=0.25,
        # epoch 104). Mechanism: logcosh gradient = tanh(e) saturates at ±1 for large
        # overshoots, so the model cannot self-correct once it tips into the sinh blowup
        # regime. 128 filters × 47 input channels provides excess capacity that amplifies
        # this trap. 64 remains under surveillance; 32 is the confirmed safe base.
        "num_filters": {"values": [32, 64]},
        # dilation_base: Fixed at 2 (standard exponential dilation, Bai et al. 2018).
        # RF formula below assumes d=2 — do not sweep without updating the RF table.
        "dilation_base": {"values": [2]},
        # num_layers: Controls receptive field. RF = 1 + (k-1)×Σ_{i=0}^{L-1}(d^i).
        # Two constraints: RF ≥ ocl=36 (full forecast horizon conditioned on past)
        # and RF ≤ icl (beyond icl is zero-padded peace — tolerable but wasteful).
        # k=3, d=2:  4 layers RF=31  (✗ RF<ocl),  5 layers RF=63  (✓),  6 layers RF=127
        # k=5, d=2:  4 layers RF=61  (✓ RF≥ocl),  5 layers RF=125 (✓),  6 layers RF=253
        # num_layers=4 removed (k=3): RF=31 < ocl=36, model extrapolates blind on
        # steps 32-36 regardless of icl.
        "num_layers": {"values": [5]},
        # weight_norm: Fixed True — constrains filter magnitudes to 1 by construction.
        # Without it, activation spikes from rare non-zero events propagate unboundedly
        # through 5 dilated conv layers, causing the blowups observed in v3 sweep.
        "weight_norm": {"values": [True]},
        "use_reversible_instance_norm": {"values": [True]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        # Dropout: TCN applies spatial dropout between conv layers.
        # Floor lowered to 0.15 vs previous logcosh sweeps: clip=1.0 + weight_norm
        # together guard against blowup, so high dropout is no longer needed as a
        # blowup prevention mechanism. Lower dropout allows 32-filter runs to retain
        # enough conflict-signal routing for Hyperband survival at epoch 25.
        # 0.35 ceiling maintained to prevent capacity overfitting on the 10% non-zero
        # conflict series — Bayesian sweep resolves the tradeoff.
        "dropout": {"values": [0.15, 0.25, 0.35]},
        # ==============================================================================
        # LOSS FUNCTION: SpotlightLossLogcosh
        # ==============================================================================
        "loss_function": {"values": ["SpotlightLossLogcosh"]},
        "non_zero_threshold": {"values": [0.88]}, 
        # delta: multi-resolution spectral weight. DC bin masked.
        "delta": {"values": [0.08]}, 
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config