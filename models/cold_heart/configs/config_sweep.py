def get_sweep_config():
    """
    meow
    """
    sweep_config = {
        "method": "bayes",
        "name": "new_rules_nbeats_spotlight_lrop_20260503_round2",
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 30,
            "eta": 2,
        },
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        # icl=48: 4yr context. N-BEATS flattens the full input window to one vector,
        # so larger icl increases the non-zero fraction of that vector and gives the
        # FC layers more conflict signal to compress. icl=36 gives no lookback
        # advantage over the output horizon — conflict persistence requires more context.
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
        "batch_size": {"values": [128]},
        "n_epochs": {"values": [300]},
        # ESP=65: covers 2+ full CAWR cycles (T_0=25). With CAWR restarts every 25 epochs,
        # ESP must not fire during a warm-restart lr trough.
        "early_stopping_patience": {"values": [50]},
        # min_delta=0.0: delegate noise filtering to CAWR's natural lr oscillation.
        "early_stopping_min_delta": {"values": [0.0]},
        "force_reset": {"values": [True]},
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        "lr": {"values": [5e-4]},
        "weight_decay": {"values": [1e-4, 1e-3]},
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["ReduceLROnPlateau"]},
        "lr_scheduler_factor": {"values": [0.7]},
        "lr_scheduler_patience": {"values": [15]},
        "lr_scheduler_min_lr": {"values": [1e-6]},
        "lr_scheduler_kwargs": {"values": [{"mode": "min", "factor": 0.7, "patience": 15, "min_lr": 1e-6, "threshold": 0.50, "threshold_mode": "abs", "cooldown": 5}]},
        "gradient_clip_val": {"values": [3.0, 5.0]},
        # ==============================================================================
        # SCALING
        # ==============================================================================
        "feature_scaler": {"values": [None]},
        "target_scaler": {"values": ["AsinhTransform"]},  # log1p(x): log-compresses targets, expm1 inverse
        "feature_scaler_map": {
            "values": [{
                "AsinhTransform->StandardScaler": [
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
        # N-BEATS ARCHITECTURE
        # ==============================================================================
        # generic_architecture: True uses generic basis (learnable), False
        # uses interpretable trend+seasonality decomposition. Generic is
        # more flexible for conflict data which lacks clean seasonality.
        "generic_architecture": {"values": [True]},
        # num_stacks: Number of stacks. Each stack processes the residual
        # from the previous. 2 is standard for generic, more adds capacity.
        # num_stacks: Fixed at 2. 3 adds ~50% params with marginal gain for
        # conflict data — N-BEATS residual path saturates fast. Saves search space.
        "num_stacks": {"values": [2]},
        # num_blocks: Locked at 2. num_blocks=1 is confirmed capacity-limited —
        # cannot route amplified β gradient into per-country differentiation.
        # Two sweep runs with num_blocks=1 both underpredict events at ~41-46%
        # despite different β values.
        "num_blocks": {"values": [2]},
        # num_layers: Fixed at 3. nl=3 outperformed nl=4 in sweep data
        # (MSLE 0.54 vs 0.63 at same width). Deeper blocks overfit ~200 series.
        "num_layers": {"values": [3]},
        # layer_widths: 256 validated. Adding 512 — N-BEATS is purely FC,
        # extra depth capacity helps on conflict series with layer_widths=512
        # remaining safe at ed=20-24 (basis matrix stays low-rank vs widths).
        "layer_widths": {"values": [128, 256, 512]},
        # expansion_coefficient_dim: basis rank in R^36.
        # ed=20 won both top runs (MSLE 0.416/0.418) vs ed=16 (MSLE 0.419).
        # ed=24 (67% of output_chunk_length): richer temporal mixture without full-rank overfitting.
        # ed=12/16 confirmed suboptimal — removed.
        "expansion_coefficient_dim": {"values": [8, 16, 24]},
        # trend_polynomial_degree: Only used in interpretable mode.
        # Included for completeness; irrelevant when generic=True.
        "trend_polynomial_degree": {"values": [2]},
        # activation: ReLU is N-BEATS paper default.
        "activation": {"values": ["GELU", "ReLU"]},
        # use_reversible_instance_norm: True for SpotlightLoss.
        # SpotlightLoss DC/AC decomposition makes RevIN safe — shape gradient
        # sums to zero per series, preventing DC offset amplification through
        # RevIN's denormalisation ŷ = ẑ·σ + μ.
        "use_reversible_instance_norm": {"values": [True]},
        "use_static_covariates": {"values": [True]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        # Dropout: N-BEATS is a deep MLP — moderate dropout needed for
        # ~200 series. Paper uses 0.0 but they had much more data.
        "dropout": {"values": [0.20, 0.30]},
        # ==============================================================================
        # LOSS FUNCTION: SpotlightLoss v36 (DRO)
        # ==============================================================================
        "loss_function": {"values": ["SpotlightLoss"]},
        "non_zero_threshold": {"values": [0.88]}, 
        # delta: multi-resolution spectral weight. DC bin masked.
        # "delta": {"distribution": "uniform", "min": 0.05, "max": 0.15},
        "delta": {"values": [0.075]},
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ====================================== ========================================
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config
