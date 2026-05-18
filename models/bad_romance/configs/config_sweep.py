def get_sweep_config():
    """
    meow
    """
    sweep_config = {
        "method": "bayes",
        "name": "smol_cat_tide_spotlight_lrop_20260503_round2",
        "early_terminate": {"type": "hyperband", "min_iter": 30, "eta": 2},
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        "input_chunk_length": {"values": [48]},
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
        # Batch size: Fixed at 64. MAAT magnitude weighting + hurdle produces
        # heterogeneous per-sample gradients — need enough samples for stable
        # batch gradient estimates. Country-level has fewer series than PGM,
        # so 64 is appropriate.
        "batch_size": {"values": [128]},
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [50]},
        "early_stopping_min_delta": {"values": [0.0]},
        "force_reset": {"values": [True]},
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        "lr": {"values": [5e-4]},
        "weight_decay": {"values": [1e-4]},
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
                    "lr_wdi_sm_pop_netm", "lr_acled_sb", "lr_acled_sb_count",
                    "lr_acled_os", 
                    
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
        # TiDE ARCHITECTURE
        # ==============================================================================
        # Country-month: fewer series (~200) but richer temporal structure.
        # Need sufficient capacity to model diverse country trajectories.
        "num_encoder_layers": {"values": [1]},
        # Decoder layers: 2 or 3. Country series are smoother than PGM —
        # 2 layers may suffice, but 3 gives more capacity for diverse patterns.
        "num_decoder_layers": {"values": [2]},  # Fixed: depth rarely decisive vs width. Saves search space.
        # decoder_output_dim: Dimensionality of the decoder output before
        # the temporal decoder. 32-64 is typical; 16 is the Darts default.
        "decoder_output_dim": {"values": [16, 32, 64, 128]},
        # hidden_size: SWEPT. Country-level needs capacity for ~200 diverse
        # trajectories. 256 is minimum viable, 512 gives headroom.
        "hidden_size": {"values": [256, 512, 768]}, # removed 256
        # temporal_width_past: Width of past covariate projection output.
        # 4 is the paper default (financial data, fewer features). With 41 monthly
        # conflict covariates, 4 is a 10:1 bottleneck — most covariate signal is
        # destroyed before the encoder sees it. Minimum viable is 12 (annual cycle);
        # 24 gives the encoder enough bandwidth for structural+conflict feature clusters.
        "temporal_width_past": {"values": [12]},
        # temporal_width_future: Width of future covariate projection output.
        # Larger values capture richer future covariate interactions.
        "temporal_width_future": {"values": [36]},  # Fixed: matches output_chunk_length. 48 adds combos without clear benefit.
        # temporal_decoder_hidden: Width of the temporal decoder MLP.
        # Needs enough capacity to map decoder output to final predictions.
        "temporal_decoder_hidden": {"values": [256, 512]}, # removed 128
        # temporal_hidden_size_past: Hidden layer width in past covariate
        # projection ResBlock. Must scale with hidden_size — at 64 the ResBlock
        # output merges into a 512-wide encoder via a near-rank-deficient projection
        # (W∈ℝ^{512×12}), wasting encoder capacity. 128/256 keeps the ratio ≤ 4:1.
        "temporal_hidden_size_past": {"values": [128]},
        # temporal_hidden_size_future: Same scaling logic as past.
        "temporal_hidden_size_future": {"values": [128]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        "use_layer_norm": {"values": [True]},
        # Dropout: Country-level has fewer training windows per series.
        # Slightly higher dropout ceiling to prevent overfitting on ~200 series.
        "dropout": {"values": [0.15]},
        "use_static_covariates": {"values": [True]},
        # RevIN on: SpotlightLoss DC/AC decomposition zeroes out per-series shape
        # gradients (Σ ∂L_shape/∂ŷᵢ = 0), preventing DC offset amplification through
        # RevIN denormalisation ŷ = ẑ·σ + μ. Safe even for sparse peace series.
        "use_reversible_instance_norm": {"values": [True]},
        "loss_function": {"values": ["SpotlightLoss"]},
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