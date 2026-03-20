def get_sweep_config():
    """
    N-HiTS + SpotlightLoss Sweep Configuration
    ============================================

    Data: ~200 country time series, 68 features (conflict + WDI + V-Dem),
    3 targets (lr_ged_sb/ns/os), 86-94% zeros, 36-month horizon.

    N-HiTS is feedforward (no BPTT) — SpotlightLoss alpha can be used freely.
    Same loss config approach as smol_cat (TiDE), adapted for N-HiTS architecture.

    Architecture (Challu et al. 2022):
    - 2 stacks: structural trends (pooled) + monthly dynamics (raw)
    - MaxPool preserves spike magnitudes in zero-inflated data
    - Hierarchical interpolation combines coarse + fine forecasts
    """

    sweep_config = {
        "method": "bayes",
        "name": "revolving_door_nhits_spotlight_v3_msle",
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
        # N-HiTS paper uses LR ~1e-3 but SpotlightLoss cosh amplification
        # injects higher gradient variance — keep ceiling conservative.
        "lr": {
            "distribution": "log_uniform_values",
            "min": 5e-5,
            "max": 5e-4,
        },
        "weight_decay": {"values": [5e-6]},
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["CosineAnnealingWarmRestarts"]},
        "lr_scheduler_T_0": {"values": [30]},
        # T_mult=2: progressive lengthening (30→60→120) lets the model
        # settle into finer optima in later cycles.
        "lr_scheduler_T_mult": {"values": [2]},
        "lr_scheduler_eta_min": {"values": [1e-6]},
        # Best run hit ceiling at 3.0. With alpha~0.6, cosh(0.6*9)≈45x
        # amplification needs gradient headroom. Drop 1.0 (too restrictive).
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
        "num_stacks": {"values": [2]},
        "num_blocks": {"values": [1]},
        "num_layers": {"values": [2]},
        # layer_widths: N-HiTS paper default is 512. 256 also viable for
        # ~200 series. Simple FC architecture handles wide layers well.
        "layer_widths": {"values": [256, 512]},
        # pooling_kernel_sizes: Multi-scale temporal aggregation.
        # Format: [[stack1_block1], [stack2_block1]]
        # Option 1: [4,1] — stack1 pools quarterly trends, stack2 raw monthly
        # Option 2: [6,1] — stack1 pools semi-annual, stack2 raw monthly
        "pooling_kernel_sizes": {"values": [[[4], [1]], [[6], [1]]]},
        # n_freq_downsample: Output resolution per stack.
        # Stack1 at 6: 36/6 = 6 basis functions (slow structural trends)
        # Stack2 at 1: 36/1 = 36 basis functions (monthly detail)
        "n_freq_downsample": {"values": [[[6], [1]]]},
        # MaxPool preserves spike magnitudes — for zero-inflated data
        # where AvgPool dilutes the rare non-zero signals.
        "max_pool_1d": {"values": [True]},
        "activation": {"values": ["ReLU"]},
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
        "use_reversible_instance_norm": {"values": [False]},
        # ==============================================================================
        # LOSS FUNCTION: SpotlightLoss
        # ==============================================================================
        # N-HiTS is feedforward — no BPTT attenuation. Same SpotlightLoss
        # parameter philosophy as smol_cat (TiDE): alpha can be used freely
        # for cosh magnitude weighting.
        "loss_function": {"values": ["SpotlightLoss"]},
        # ── alpha (magnitude expansion rate) ──────────
        # Best run: 0.618, well-centered. N-HiTS is feedforward — cosh
        # is safe. Widen floor to let Bayes explore less amplification.
        "alpha": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 0.9,
        },
        # ── beta (asymmetry strength) ─────────────────
        # Under-predicting real conflict costs (1+beta)x more than over-predicting.
        # Conservative range because alpha already heavily favors rare events.
        "beta": {
            "distribution": "uniform",
            "min": 0.3,
            "max": 0.7,
        },
        # ── kappa (sigmoid sharpness) ─────────────────
        # Best run: 13.34, near ceiling. N-HiTS's MaxPool preserves spike
        # magnitudes so sharp FP/FN switching works. Raise ceiling, drop
        # the low end (kappa<10 is too smooth to help discrimination).
        "kappa": {
            "distribution": "uniform",
            "min": 8.0,
            "max": 16.0,
        },
        # ── delta (huber threshold) ───────────────────
        # Best run: 1.377, hit 92% of ceiling. Higher delta gives the coarse
        # stack full quadratic gradient for larger errors — MaxPool +
        # hierarchical interpolation smooth the noise, so it's safe.
        # Center around best run with room to explore upward.
        "delta": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 2.0,
        },
        # ── gamma (temporal weight) ───────────────────
        # Best run: 0.059, at floor. N-HiTS's coarse stack already produces
        # smooth forecasts via low-frequency basis functions — extra temporal
        # penalty is redundant and constrains the fine stack.
        "gamma": {
            "distribution": "uniform",
            "min": 0.05,
            "max": 0.4,
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