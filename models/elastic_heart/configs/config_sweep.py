def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    Optimized for TSMixerModel on zero-inflated conflict fatalities data.
    
    TSMixer Architecture Notes:
    - MLP-based mixer alternates time-mixing and feature-mixing layers
    - Efficient alternative to attention-based models
    - Feed-forward size controls model capacity
    - Normalization type crucial for zero-inflated distributions
    - Fewer parameters than Transformers, faster training
    
    Returns:
    - sweep_config (dict): Configuration for hyperparameter sweeps.
    """

    sweep_config = {
        'method': 'bayes',
        'name': 'elastic_heart_tsmixer_v1_mtd',
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 20,
            'eta': 2
        },
        'metric': {
            'name': 'time_series_wise_mtd_mean_sb',
            'goal': 'minimize'
        },
    }

    parameters = {
        # ============== TEMPORAL CONFIGURATION ==============
        'steps': {'values': [[*range(1, 36 + 1)]]},
        'input_chunk_length': {'values': [36, 48, 60]},
        'output_chunk_length': {'values': [36]},

        "output_chunk_shift": {"values": [0]},  # No gap between input and forecast
        "random_state": {"values": [67]},  # Reproducibility
        "mc_dropout": {"values": [True]},  # Monte Carlo dropout for uncertainty

        # ============== TRAINING BASICS ==============
        'batch_size': {'values': [1024, 2048, 4096]},
        'n_epochs': {'values': [150]},
        'early_stopping_patience': {'values': [20]},
        'early_stopping_min_delta': {'values': [0.0001]},
        'force_reset': {'values': [True]},

        # ============== OPTIMIZER / SCHEDULER ==============
        "lr": {
            "distribution": "log_uniform_values",
            "min": 5e-5,
            "max": 1e-3,
        },
        "weight_decay": {"values": [0]},
        # lr_scheduler: ReduceLROnPlateau configuration
        # - factor=0.5: Halve LR when stuck (standard, well-tested)
        # - patience=8: Wait 8 epochs before reducing
        # - min_lr=1e-6: Floor prevents LR from becoming negligible
        "lr_scheduler_factor": {"values": [0.5]},
        "lr_scheduler_patience": {"values": [8]},
        "lr_scheduler_min_lr": {"values": [1e-6]},

        # gradient_clip_val: Maximum gradient norm
        # - Prevents exploding gradients
        # - TSMixer has stable gradients due to simple MLP architecture
        # - Value 1.5 is conservative for [0,1] scaled data
        "gradient_clip_val": {"values": [1.5]},

        # ============== SCALING ==============
        "feature_scaler": {"values": [None]},

        # target_scaler: AsinhTransform->MinMaxScaler
        # - Asinh: Handles zeros, compresses extremes
        #   * asinh(0) = 0 (no log(0) issues)
        #   * Linear near 0, logarithmic for large values
        # - MinMax: Bounds to [0,1] for stable gradients
        "target_scaler": {"values": ["AsinhTransform->MinMaxScaler"]},

        # feature_scaler_map: Per-feature scaling based on distribution
        "feature_scaler_map": {
            "values": [
                {
                    # AsinhTransform->MinMaxScaler: Zero-inflated and right-skewed
                    # Features with many zeros and occasional extreme values
                    "AsinhTransform->MinMaxScaler": [
                        # Conflict fatality counts (zero-inflated, extreme outliers)
                        "lr_ged_sb", "lr_ged_ns", "lr_ged_os",
                        "lr_acled_sb", "lr_acled_sb_count", "lr_acled_os",
                        "lr_ged_sb_tsum_24",  # 24-month cumulative
                        "lr_splag_1_ged_sb", "lr_splag_1_ged_os", "lr_splag_1_ged_ns",
                        # Economic indicators (GDP spans orders of magnitude)
                        "lr_wdi_ny_gdp_mktp_kd", "lr_wdi_nv_agr_totl_kn",
                        "lr_wdi_sm_pop_netm", "lr_wdi_sm_pop_refg_or",
                        # Mortality rates (positive, skewed)
                        "lr_wdi_sp_dyn_imrt_fe_in",
                        # Token counts from text (zero-inflated)
                        "lr_topic_tokens_t1", "lr_topic_tokens_t1_splag",
                    ],
                    # MinMaxScaler: Bounded or roughly symmetric features
                    # Already in reasonable ranges, just need [0,1] normalization
                    "MinMaxScaler": [
                        # WDI percentages (0-100 scale)
                        "lr_wdi_sl_tlf_totl_fe_zs", "lr_wdi_se_enr_prim_fm_zs",
                        "lr_wdi_sp_urb_totl_in_zs", "lr_wdi_sh_sta_maln_zs",
                        "lr_wdi_sh_sta_stnt_zs", "lr_wdi_dt_oda_odat_pc_zs",
                        "lr_wdi_ms_mil_xpnd_gd_zs",
                        # V-Dem indices (already 0-1)
                        "lr_vdem_v2x_horacc", "lr_vdem_v2xnp_client", "lr_vdem_v2x_veracc",
                        "lr_vdem_v2x_divparctrl", "lr_vdem_v2xpe_exlpol", "lr_vdem_v2x_diagacc",
                        "lr_vdem_v2xpe_exlgeo", "lr_vdem_v2xpe_exlgender", "lr_vdem_v2xpe_exlsocgr",
                        "lr_vdem_v2x_ex_party", "lr_vdem_v2x_genpp", "lr_vdem_v2xeg_eqdr",
                        "lr_vdem_v2xcl_prpty", "lr_vdem_v2xeg_eqprotec", "lr_vdem_v2x_ex_military",
                        "lr_vdem_v2xcl_dmove", "lr_vdem_v2x_clphy", "lr_vdem_v2x_hosabort",
                        "lr_vdem_v2xnp_regcorr",
                        # Topic thetas (probability proportions, sum to ~1)
                        "lr_topic_ste_theta0", "lr_topic_ste_theta1", "lr_topic_ste_theta2",
                        "lr_topic_ste_theta3", "lr_topic_ste_theta4", "lr_topic_ste_theta5",
                        "lr_topic_ste_theta6", "lr_topic_ste_theta7", "lr_topic_ste_theta8",
                        "lr_topic_ste_theta9", "lr_topic_ste_theta10", "lr_topic_ste_theta11",
                        "lr_topic_ste_theta12", "lr_topic_ste_theta13", "lr_topic_ste_theta14",
                        "lr_topic_ste_theta0_stock_t1_splag", "lr_topic_ste_theta1_stock_t1_splag",
                        "lr_topic_ste_theta2_stock_t1_splag", "lr_topic_ste_theta3_stock_t1_splag",
                        "lr_topic_ste_theta4_stock_t1_splag", "lr_topic_ste_theta5_stock_t1_splag",
                        "lr_topic_ste_theta6_stock_t1_splag", "lr_topic_ste_theta7_stock_t1_splag",
                        "lr_topic_ste_theta8_stock_t1_splag", "lr_topic_ste_theta9_stock_t1_splag",
                        "lr_topic_ste_theta10_stock_t1_splag", "lr_topic_ste_theta11_stock_t1_splag",
                        "lr_topic_ste_theta12_stock_t1_splag", "lr_topic_ste_theta13_stock_t1_splag",
                        "lr_topic_ste_theta14_stock_t1_splag",
                        # Population growth (small range, can be negative)
                        "lr_wdi_sp_pop_grow",
                    ],
                }
            ]
        },

        # ============== TSMIXER ARCHITECTURE ==============
        'hidden_size': {'values': [256, 512]},  # Main hidden dimension
        'ff_size': {'values': [512, 1024]},  # Feed-forward expansion size
        'num_blocks': {'values': [2, 3, 4]},  # Number of mixer blocks
        "dropout": {"values": [0.05, 0.15]},
        'norm_type': {'values': ['LayerNorm']},
        'normalize_before': {'values': [True, False]},  # Pre-normalization typically better
        'activation': {'values': ['ReLU', 'GELU']},
        'use_static_covariates': {'values': [True]},
        'use_reversible_instance_norm': {'values': [True, False]},  # Helps with distribution shift

        # ============== LOSS FUNCTION ==============
        # WeightedPenaltyHuberLoss: Custom loss for zero-inflated rare event forecasting
        # Weight multiplication logic:
        # - True Negative (zero→zero): weight = 1.0 (baseline)
        # - False Positive (zero→non-zero): weight = false_positive_weight
        # - True Positive (non-zero→non-zero): weight = non_zero_weight
        # - False Negative (non-zero→zero): weight = non_zero_weight × fn_weight
        'loss_function': {'values': ['WeightedPenaltyHuberLoss']},
        
        # zero_threshold: Scaled value below which predictions count as "zero"
        # - After AsinhTransform->MinMaxScaler, 1 fatality ≈ 0.11
        # - Range 0.08-0.15 spans ~0.5-2 fatalities threshold
        "zero_threshold": {
            "distribution": "uniform",
            "min": 0.08,
            "max": 0.15,
        },

        # delta: Huber loss transition point (L2 inside delta, L1 outside)
        # - Range 0.7-1.0 gives nearly pure L2 behavior for [0,1] scaled data
        # - Full L2 maximizes gradient signal from every error
        "delta": {
            "distribution": "uniform",
            "min": 0.7,
            "max": 1.0,
        },

        # non_zero_weight: Multiplier for non-zero actual values
        # - PINNED at 10.0 to reduce search dimensions
        # - Conflicts contribute 10x more to loss than zeros
        "non_zero_weight": {"values": [10.0]},

        # false_positive_weight: Multiplier when predicting non-zero for actual zero
        # - Range 1.0-1.5: Neutral to slight penalty for false alarms
        # - Prevents peaceful countries from spiking
        "false_positive_weight": {
            "distribution": "uniform",
            "min": 1.0,
            "max": 1.5,
        },

        # false_negative_weight: Additional penalty for missing actual conflicts
        # - Applied on top of non_zero_weight: FN = 10 × fn_weight = 20-40x baseline
        # - Penalizes missing conflicts without causing over-hedging
        "false_negative_weight": {
            "distribution": "uniform",
            "min": 2.0,
            "max": 8.0,
        },
    }

    sweep_config['parameters'] = parameters
    return sweep_config