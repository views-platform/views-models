
def get_sweep_config():
    """N-HiTS + SpotlightLoss sweep configuration."""

    sweep_config = {
        "method": "bayes",
        "name": "revolving_door_nhits_spotlight_lrop_20260503_round2",
        "early_terminate": {"type": "hyperband", "min_iter": 30, "eta": 2},
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
        "batch_size": {"values": [128]},
        "n_epochs": {"values": [300]},
        # ESP=65: covers 2+ full CAWR cycles (T_0=25). Warm-restart valleys can cause
        # brief apparent plateaus — ESP must not fire during the trough.
        "early_stopping_patience": {"values": [50]},
        "early_stopping_min_delta": {"values": [0.0]},
        "force_reset": {"values": [True]},
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        "lr": {"values": [5e-4]},
        # wd=1e-3: confirmed best in runs with layer_widths=128/256.
        # wd=1e-4: included for wider (512) configs where stronger wd may overregularise.
        # Note: effective wd per step = lr × wd. At lr=5e-4, wd=1e-3 → 5e-7/step.
        "weight_decay": {"values": [1e-4, 1e-3]},
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["ReduceLROnPlateau"]},
        "lr_scheduler_factor": {"values": [0.7]},
        "lr_scheduler_patience": {"values": [15]},
        "lr_scheduler_min_lr": {"values": [1e-6]},
        "lr_scheduler_kwargs": {"values": [{"mode": "min", "factor": 0.7, "patience": 15, "min_lr": 1e-6, "threshold": 0.50, "threshold_mode": "abs", "cooldown": 5}]},
        "gradient_clip_val": {"values": [1.0, 3.0, 5.0]},
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
        # N-HiTS ARCHITECTURE
        # ==============================================================================
        # 3 stacks: coarse + intermediate + fine.
        # Pooling kernel rationale (icl=48):
        #   kernel=6  → 48/6  = 8 half-year groups (coarse trends)
        #   kernel=3  → 48/3  = 16 quarterly groups (conflict-cycle patterns)
        #   kernel=1  → 48 raw monthly steps        (fine stack always unpooled)
        # n_freq_downsample rationale (ocl=36, independent of icl):
        #   fd=6 → 36/6 = 6 basis functions (slow trends, enough to localise errors)
        #   fd=2 → 36/2 = 18 basis functions (sub-quarterly detail)
        #   fd=1 → 36/1 = 36 basis functions (full monthly)
        # NOTE: previous config used fd=12 (3 basis) for coarse stack. With
        # AsinhTransform, a single overshot coefficient was interpolated across
        # 12 output steps → sinh amplified it to billions. 6 basis functions give
        # the coarse stack enough local freedom to self-correct while still
        # capturing slow structural trends.
        "num_stacks": {"values": [3]},
        "pooling_kernel_sizes": {"values": [[[6],[3],[1]]]},
        "n_freq_downsample": {"values": [[[6],[2],[1]]]},  # Darts requires fd[-1][-1]==1. Fine stack is full-rank; regularization via dropout/wd/width instead.
        "max_pool_1d": {"values": [False]},  # True causes spike propagation via max-pooling on sparse conflict data (confirmed Run 2)
        "activation": {"values": ["GELU"]},
        # num_blocks/num_layers/layer_widths: Darts defaults are 1 block, 2 layers,
        # 512 width per stack. num_blocks fixed at 1: zero-inflation is addressed by
        # SpotlightLoss+AsinhTransform, not within-stack depth. num_blocks=2 requires
        # paired pooling/downsampling tuples per stack and overfits sparse peace series.
        # Capacity is swept via layer_widths instead.
        # num_blocks fixed at 1: Bayes sweeps pick num_blocks and
        # pooling_kernel_sizes independently. num_blocks=2 requires paired tuples
        # per stack (e.g. [[6,6],[3,3],[1,1]]) — incompatible with [[6],[3],[1]].
        # Architecturally: within-stack depth is redundant here; SpotlightLoss +
        # AsinhTransform handle zero-inflation. Capacity is swept via layer_widths.
        "num_blocks": {"values": [1]},
        "num_layers": {"values": [2, 3]},
        # layer_widths: w=128 confirmed viable. w=512 produced event_ratio=0.71
        # vs 0.25 for w=128 in a prior run with hybrid RINorm — 4× capacity allows
        # the model to learn high-conflict dynamics instead of collapsing to
        # the peace-series mean. w=256 is the midpoint.
        "layer_widths": {"values": [512, 768, 1024]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        # Dropout: 0.30 confirmed best at w=128. 0.35 accompanies w=512 in the
        # prior winning run — wider network needs stronger dropout to avoid overfitting
        # ~200 country-level series.
        "dropout": {"values": [0.20, 0.30, 0.35]},
        "use_static_covariates": {"values": [True]},
        # RevIN on: SpotlightLoss+AsinhTransform keeps outputs bounded; RevIN normalises
        # per-series mean/variance before encoding, improving convergence across heterogeneous
        # conflict intensities (peaceful vs. high-casualty series).
        "use_reversible_instance_norm": {"values": [True]},
        # ==============================================================================
        # LOSS FUNCTION: SpotlightLoss
        # ==============================================================================
        "loss_function": {"values": ["SpotlightLoss"]},
        "non_zero_threshold": {"values": [0.88]}, 
        # delta: multi-resolution spectral weight. DC bin masked.
        "delta": {"values": [0.075]},
        # ModelCatalog builds the encoder dict from this flag at model-build
        # time, selecting functions based on config["level"] — JSON-safe.
        "use_cyclic_encoders": {"values": [True]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config