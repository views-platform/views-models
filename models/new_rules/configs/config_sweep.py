def get_sweep_config():
    """
    meow
    """
    sweep_config = {
        "method": "bayes",
        "name": "new_rules_nbeats_shadow_20260504",
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
        "steps": {"values": [[*range(1, 36 + 1)]]},
        "input_chunk_length": {"values": [36]},
        "output_chunk_shift": {"values": [0]},
        "random_state": {"values": [67]},
        "output_chunk_length": {"values": [36]},
        "optimizer_cls": {"values": ["AdamW"]},
        "mc_dropout": {"values": [False]},
        "num_samples": {"values": [1]},
        "n_jobs": {"values": [-1]},
        # ==============================================================================
        # TRAINING
        # ==============================================================================
        "batch_size": {"values": [128]},
        "n_epochs": {"values": [300]},
        # ESP=35: allows ~4 LR reductions (patience=8 each) before triggering.
        # Each RLROP firing gives the optimizer a reset opportunity; 35 epochs of
        # continuous stagnation despite all reductions is a reliable stop signal.
        # Hyperband (min_iter=15) is the primary fast-kill for clearly bad runs.
        "early_stopping_patience": {"values": [35]},
        "early_stopping_min_delta": {"values": [0.001]},
        "force_reset": {"values": [True]},
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        "lr": {"values": [5e-4, 2e-4]},
        # wd=1e-3: at lr=5e-4, effective wd/step = 5e-7 — sufficient for ~200 series.
        "weight_decay": {"values": [1e-3, 1e-4]},
        # ==============================================================================
        # LR SCHEDULER: ReduceLROnPlateau
        # RLROP on val_loss: val_loss (test partition, frozen scalers) is significantly
        # smoother than train_loss on conflict batches, so RLROP's plateau detection
        # is reliable here. factor=0.5 (halve LR) is gentle enough for a noisy val
        # signal on ~200 series. patience=8: allows ~4 LR drops within the ESP=35
        # window (8, 16, 24, 32 epochs of stagnation) before early stopping triggers —
        # each drop gives the optimizer a fresh shot before committing to stop.
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["ReduceLROnPlateau"]},
        "lr_scheduler_factor": {"values": [0.5]},
        "lr_scheduler_patience": {"values": [8]},
        "lr_scheduler_min_lr": {"values": [1e-6]},
        "lr_scheduler_kwargs": {"values": [{"mode": "min", 
                                            "factor": 0.5, 
                                            "patience": 8, 
                                            "min_lr": 1e-6, 
                                            "threshold": 0.01, 
                                            "threshold_mode": "rel", 
                                            "cooldown": 3}]},
        # TiDE: skip path + unconstrained output → tight clipping. Pinned to
        # remove three-way interaction with weight_decay and dropout.
        "gradient_clip_val": {"values": [1.0, 3.0, 5.0]},
        # ==============================================================================
        # SCALING
        # ==============================================================================
        "feature_scaler": {"values": [None]},
        "target_scaler": {"values": ["AsinhTransform"]},
        "feature_scaler_map": {
            "values": [{
                "AsinhTransform->StandardScaler": [
                    # Macro volumes: 5+ order-of-magnitude cross-country difference.
                    # StandardScaler alone produces 50σ activations for large economies.
                    "lr_wdi_ny_gdp_mktp_kd",
                    "lr_wdi_nv_agr_totl_kn",
                    # Zero-inflated with heavy right tail.
                    "lr_wdi_sm_pop_refg_or",
                
                    # Signed, heavy tails both directions.
                    "lr_wdi_sm_pop_netm",
                
                    # Infant mortality: Finland ~1.5, Chad ~90 — ~2 orders of magnitude.
                    # Strongly conflict-predictive; tail compression is essential.
                    "lr_wdi_sp_dyn_imrt_fe_in",
                    
                    # Rates and ratios without extreme skew or multi-order range.
                    # Pop growth: near-normal, signed.
                    "lr_wdi_sp_pop_grow",
                    # Female labour: bell-shaped ~35–50%.
                    # Enrolment ratio: clusters near 100. Urbanisation: near-uniform 10–90%.
                    "lr_wdi_sl_tlf_totl_fe_zs",
                    "lr_wdi_se_enr_prim_fm_zs",
                    "lr_wdi_sp_urb_totl_in_zs",

                    # Stunting/malnutrition 2–55%: asinh compresses 27× range to 3.3×.
                    # Right-skewed and conflict-predictive — tail signal matters.
                    "lr_wdi_sh_sta_stnt_zs",
                    "lr_wdi_sh_sta_maln_zs",
                    "lr_wdi_dt_oda_odat_pc_zs",

                    # Military % GDP: median ~1.5%, outliers at 10–25% (Saudi, NK).
                    # StandardScaler alone → 5–10σ activations for outlier countries.
                    "lr_wdi_ms_mil_xpnd_gd_zs",
                ],
                "AsinhTransform->MaxAbsScaler": [
                    # Conflict counts, spatial lags, deltas: zero-inflated,
                    # 2–5 orders of magnitude cross-country range. asinh compresses
                    # the tail; MaxAbsScaler maps to [0,1] preserving zero=0 anchor
                    # and full proportional tail discrimination (no mean-shift).
                    "lr_splag_1_ged_sb",
                    "lr_splag_1_ged_ns",
                    "lr_splag_1_ged_os",
                    "lr_ged_ns",
                    "lr_ged_os",
                    "lr_ged_sb_delta",
                    "lr_ged_ns_delta",
                    "lr_ged_os_delta",
                    "lr_acled_sb",
                    "lr_acled_sb_count",
                    "lr_acled_os",
                ],
                "MinMaxScaler": [
                    # V-Dem [0,1] IRT indices: IRT construction places empirical range
                    # near the full [0,1] interval across ~200 countries. Many are
                    # bimodal or heavily skewed (e.g. v2x_ex_military: most near 0,
                    # some near 1). StandardScaler destroys this structure; MinMaxScaler
                    # maps to [0,1] matching the index construction.
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
        "loss_function": {"values": ["SpotlightLossLogcosh"]},
        "non_zero_threshold": {"values": [0.88]}, 
        # delta: multi-resolution spectral weight. DC bin masked.
        # "delta": {"distribution": "uniform", "min": 0.05, "max": 0.15},
        "delta": {"values": [0.075]},
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config
