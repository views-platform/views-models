def get_sweep_config():
    """
    """
    sweep_config = {
        "method": "bayes",
        "name": "elastic_heart_tsmixer_spotlight_lrop_20260503",
        "early_terminate": {
            "type": "hyperband",
            # CAWR T_0=25: min_iter=30 = 5 epochs post-restart-1, past the spike; comparisons at matched post-restart phase.
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
        # icl=48: fc_hist=Linear(48→36) forces temporal compression/summarization.
        # icl=36 (square): fc_hist learns to copy recent pattern forward, amplifying extremes
        #   — died at epoch 42 in sweep, confirming it.
        # icl=72: over-compresses (72→36), loses conflict persistence signal — MSLE 0.404 vs 0.378.
        # icl=48 is the confirmed sweet spot.
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
        "early_stopping_patience": {"values": [40]},
        # min_delta=0.0: delegate improvement detection entirely to RLROP abs=0.50 threshold.
        # 1e-3 min_delta was redundant noise that caused early termination.
        "early_stopping_min_delta": {"values": [0.0]},
        "force_reset": {"values": [True]},
        
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        "lr": {"values": [3e-5, 2e-4]},
        "weight_decay": {"values": [1e-4, 1e-3]},
        
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["ReduceLROnPlateau"]},
        "lr_scheduler_factor": {"values": [0.7]},
        "lr_scheduler_patience": {"values": [15]},
        "lr_scheduler_min_lr": {"values": [1e-6]},
        "lr_scheduler_kwargs": {"values": [{"mode": "min", "factor": 0.7, "patience": 15, "min_lr": 1e-6, "threshold": 0.50, "threshold_mode": "abs", "cooldown": 5}]},
        "gradient_clip_val": {"values": [2.0, 3.0, 5.0]},
        # ==============================================================================
        # SCALING
        # ==============================================================================
        "feature_scaler": {"values": [None]},
        "target_scaler": {"values": ["AsinhTransform"]},  # asinh(x): SpotlightLoss operates in asinh space. non_zero_threshold=0.88=asinh(1).
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
        # TSMIXER ARCHITECTURE
        # ==============================================================================
        # num_blocks=3 was the decisive factor in the winning run (MSLE=0.378 vs 0.404).
        # train_loss=0.283 (3 blocks) vs 0.903 (2 blocks) with same ff/hidden/lr — 3 blocks
        # give ~3× more feature compression across the static/past/future covariate streams.
        # The previous concern about "3× amplification" didn't materialize with dropout=0.35.
        "num_blocks": {"values": [2, 3]},
        # hidden=128 + ff=512: all 3 winning runs used these values (no overprediction).
        # The previous concern about ff>hidden causing amplification was from a higher-lr/
        # no-dropout regime. With dropout=0.35 and clip=2, ff=512 is safe and necessary —
        # the FC layers need capacity to mix 45 features across covariate streams.
        "hidden_size": {"values": [128]},
        "ff_size": {"values": [256, 512]},
        "normalize_before": {"values": [True]},
        "activation": {"values": ["GELU"]},
        "norm_type": {"values": ["LayerNorm"]},
        
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        # dropout=0.35 won both top runs. Values below 0.30 are underfitting for 200 series;
        # 0.15/0.20 dropped from search space as confirmed suboptimal.
        "dropout": {"values": [0.15, 0.20, 0.30]},
        "use_static_covariates": {"values": [True]},
        # RevIN disabled: log1p peace series have σ≈0 → division instability.
        # Also prevents RevIN from homogenizing temporal shapes across countries.
        "use_reversible_instance_norm": {"values": [True]},
        
        # ==============================================================================
        # STATIC COVARIATE STATS
        # ==============================================================================
        # Per-entity fingerprint stats (mu, sigma, max, trend, sparsity) are
        # injected as static covariates. Raw stats have 38,000× scale mismatch
        # with asinh model space (Syria max=500k vs asinh=13). AsinhTransform
        # compresses them to the same ~[0,14] range as model internals, making
        # the joint LayerNorm inside feature_mixing_static meaningful.
        "static_covariate_stats": {"values": [{"transform": "AsinhTransform"}]},
        
        # ==============================================================================
        # LOSS FUNCTION: PrismLoss v34
        # ==============================================================================
        "loss_function": {"values": ["SpotlightLossLogcosh"]},
        "non_zero_threshold": {"values": [0.88]}, 
        # delta: multi-resolution spectral weight. DC bin masked.
        "delta": {"values": [0.075]},
        
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config