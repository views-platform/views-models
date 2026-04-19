
def get_hp_config():
    """
    TSMixer hyperparameters for v20 SpotlightLoss test run.

    SpotlightLoss v20 — 3 stages:
      Stage 1 — truth-only weight: w(y) = 1+log_cosh(alpha*|y|)
                alpha=0.3 → max weight ~3.8× at y=11.5 (max UCDP),
                1.0× at y=0. Saturating (linear tail) — noisy 50k spike
                can't hijack shared MLP weights. NO pred-side weight.
      Stage 2 — 50/50 balanced mean across event/peace buckets.
                Prevents 90% peace cells drowning 10% event gradient.
      Stage 3 — multi-resolution STFT spectral matching (delta=0.15).
                n_fft=(6,12,24), magnitude-only (phase discarded).
                Replaces TV: catches oscillation + seasonal + slow drift.
                Phase-invariant: onset 1-mo early ≈ zero spectral penalty.

    Returns:
    - hyperparameters (dict): Training configuration dictionary.
    """

    hyperparameters = {
        # Temporal
        "steps": [*range(1, 36 + 1, 1)],
        "input_chunk_length": 72,
        "output_chunk_length": 36,
        "output_chunk_shift": 0,
        "random_state": 67,
        "time_steps": 36,  # Checksum: Must match len(steps)
        "rolling_origin_stride": 1,

        # Inference
        "num_samples": 1,
        "mc_dropout": False,
        "n_jobs": -1,

        # Training
        "batch_size": 64,
        "n_epochs": 100,
        "early_stopping_patience": 20,
        "early_stopping_min_delta": 0.0001,
        "force_reset": True,

        # Optimizer
        "optimizer_cls": "RAdam",
        "lr": 0.0003,
        "weight_decay": 0,
        # v20 spectral loss: per-cell gradient hard-bounded at w(y)×1.0 ≤ 3.8.
        # Clip at 5 for test run to avoid clipping event cells (max 3.8×).
        # Tighten to 2-3 if overprediction observed.
        "gradient_clip_val": 5,

        # LR Scheduler
        "lr_scheduler_cls": "CosineAnnealingWarmRestarts",
        "lr_scheduler_T_0": 30,
        "lr_scheduler_T_mult": 2,
        "lr_scheduler_eta_min": 1e-6,
        "lr_scheduler_kwargs": {
            "T_0": 30,
            "T_mult": 2,
            "eta_min": 1e-6,
        },
        "optimizer_kwargs": {
            "lr": 0.0003,
            "weight_decay": 0,
        },

        # Loss: SpotlightLoss v20 — truth-only 1+log_cosh(alpha*|y|) weight + multi-res spectral
        # alpha=0.3: truth at max UCDP (asinh≈11.5) → 1+log_cosh(3.45)≈3.8× (truncated inv-density)
        #            Saturating (linear growth): noisy 50k spike gets ~3.8×, not ~16×
        # delta=0.15: multi-resolution spectral matching (n_fft=6,12,24)
        #             Phase-insensitive: spike 1-mo early → ~zero penalty (Fourier shift theorem)
        #             n_fft=12 bin 1 captures 12-month annual seasonality directly
        #             Replaces TV — spectral catches oscillation + drift + seasonality
        "loss_function": "SpotlightLoss",
        "alpha": 0.3,
        "delta": 0.15,
        "non_zero_threshold": 0.88,

        # Scaling
        "feature_scaler": None,
        "target_scaler": "AsinhTransform",
        "feature_scaler_map": {
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
        },

        # TSMixer Architecture
        "num_blocks": 2,
        "hidden_size": 128,
        "ff_size": 512,
        "activation": "GELU",
        "norm_type": "LayerNorm",
        "normalize_before": True,
        "dropout": 0.25,
        "use_static_covariates": True,
        "use_reversible_instance_norm": True,

        "use_cyclic_encoders": True,

        # Prediction output format
        "prediction_format": "dataframe",
    }
    return hyperparameters
