def get_sweep_config():
    """
    meow
    """
    sweep_config = {
        "method": "bayes",
        "name": "good_life_transformer_spotlight_v2_msle",
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
        "mc_dropout": {"values": [True]},
        "detect_anomaly": {"values": [False]},
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
        # Transformers are notoriously sensitive to LR — the warmup from
        # CosineAnnealing restarts helps, but the range must be conservative.
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 3e-4,
        },
        "weight_decay": {"values": [5e-6]},
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["CosineAnnealingWarmRestarts"]},
        "lr_scheduler_T_0": {"values": [30]},
        "lr_scheduler_T_mult": {"values": [2]},
        "lr_scheduler_eta_min": {"values": [1e-6]},
        # Transformers benefit from moderate clipping — attention can
        # produce gradient spikes, but too tight kills long-range signal.
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
        # TRANSFORMER ARCHITECTURE
        # ==============================================================================
        # d_model: Embedding dimension. Constrained jointly with nhead so
        # that head_dim = d_model / nhead >= 32 for stable attention.
        # 128/4=32 is the proven minimum; 128/2=64 gives richer heads.
        "d_model": {"values": [128]},
        # nhead: 4 gives head_dim=32 (tight but stable), 2 gives 64 (rich).
        # Both valid with d_model=128. Avoids the 64/4=16 trap entirely.
        "nhead": {"values": [2, 4]},
        # num_encoder_layers: 2-3 layers. ~200 series don't need deep
        # encoders; 2 is standard, 3 adds capacity for temporal complexity.
        "num_encoder_layers": {"values": [2, 3]},
        # num_decoder_layers: Match or slightly fewer than encoder.
        # Decoder complexity should mirror encoder for balanced attention.
        "num_decoder_layers": {"values": [2, 3]},
        # dim_feedforward: FF expansion factor. 2-4x d_model.
        # 256-512 for d_model=64-128. Controls capacity of position-wise FF.
        "dim_feedforward": {"values": [256, 512]},
        # activation: Gated activations (GEGLU, SwiGLU) outperform vanilla
        # relu/gelu in recent Transformer literature (Shazeer 2020).
        "activation": {"values": ["gelu", "SwiGLU"]},
        # norm_type: LayerNorm is standard and most stable.
        "norm_type": {"values": ["LayerNorm"]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        # Dropout: Transformers with ~200 series overfit fast. 0.15 is the
        # practical floor — below that, attention memorizes training windows.
        "dropout": {
            "distribution": "uniform",
            "min": 0.15,
            "max": 0.35,
        },
        "use_reversible_instance_norm": {"values": [True, False]},
        # ==============================================================================
        # LOSS FUNCTION: SpotlightLoss
        # ==============================================================================
        "loss_function": {"values": ["SpotlightLoss"]},
        # ── alpha (cosh magnitude rate) ────────────────
        # cosh(alpha * |y|): at alpha=0.4, Ukraine (asinh≈9.9) gets ~23×
        # weight.  smol_cat best: 0.387.  Range tightly around that.
        #   0.2: mild (cosh(2.0)≈3.8×)
        #   0.4: sweet spot (~23×)  [smol_cat]
        #   0.5: strong (cosh(5.0)≈74×)
        "alpha": {
            "distribution": "uniform",
            "min": 0.2,
            "max": 0.5,
        },
        
        # ── beta (asymmetry strength) ─────────────────
        # smol_cat best: 0.236.  Zero beta caused Ukraine collapse
        # because the model freely under-predicted extreme cells.
        "beta": {
            "distribution": "uniform",
            "min": 0.1,
            "max": 0.4,
        },
        
        # ── kappa (sigmoid sharpness) ─────────────────
        # smol_cat best: 12.49.  Sharp is GOOD — clean binary
        # switch between FN/FP regimes, no mushy gradient zone.
        "kappa": {
            "distribution": "uniform",
            "min": 8.0,
            "max": 15.0,
        },
        # ── gamma (temporal weight) ───────────────────
        # smol_cat best: 0.129.  Strong temporal gradient matching
        # constrains wild discontinuities between timesteps.
        "gamma": {
            "distribution": "uniform",
            "min": 0.05,
            "max": 0.2,
        },
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config