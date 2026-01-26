def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    Optimized for TFT on zero-inflated, right-skewed conflict data at country-month level.
    
    TFT Flat Line Prevention:
    - hidden_size must be >= 32 (ideally 64-128) for meaningful attention
    - hidden_size must be divisible by num_attention_heads with quotient >= 16
    - Gradient clipping prevents attention collapse
    - Temporal encodings (add_encoders) help capture seasonality
    - Learning rate must be low enough for stable attention learning

    Returns:
    - sweep_config (dict): Configuration for hyperparameter sweeps.
    """

    sweep_config = {
        "method": "bayes",
        "name": "tft_dylan_cm_rinF",
        "early_terminate": {"type": "hyperband", "min_iter": 15, "eta": 2},
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ============== TEMPORAL CONFIGURATION ==============
        "steps": {"values": [[*range(1, 36 + 1)]]},
        "input_chunk_length": {"values": [36, 48, 60]},  # Longer context helps TFT
        "output_chunk_shift": {"values": [0]},
        
        # ============== TRAINING BASICS ==============
        # Smaller batches help with sparse/zero-inflated data
        "batch_size": {"values": [32, 64]},
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [12, 15]},
        "early_stopping_min_delta": {"values": [0.0005, 0.001]},
        "force_reset": {"values": [True]},
        
        # ============== OPTIMIZER / SCHEDULER ==============
        # TFT needs LOW learning rates to prevent attention collapse
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 5e-4,  # REDUCED: high LR causes flat predictions
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 5e-4,
        },
        "lr_scheduler_factor": {
            "distribution": "uniform",
            "min": 0.3,
            "max": 0.5,
        },
        "lr_scheduler_patience": {"values": [5, 7]},
        "lr_scheduler_min_lr": {"values": [1e-7]},
        
        # CRITICAL: Gradient clipping prevents attention explosion -> flat lines
        "gradient_clip_val": {
            "distribution": "uniform",
            "min": 0.3,
            "max": 0.8,
        },
        # Scaling and transformation
        "feature_scaler": {"values": [None]},
        "target_scaler": {
            "values": ["AsinhTransform->MinMaxScaler", "RobustScaler->MinMaxScaler"]
        },
        "feature_scaler_map": {
            "values": [
                {
                    # Zero-inflated conflict counts - asinh handles zeros and extreme spikes
                    "AsinhTransform->MinMaxScaler": [
                        "lr_ged_sb",
                        "lr_ged_ns",
                        "lr_ged_os",
                        "lr_acled_sb",
                        "lr_acled_sb_count",
                        "lr_acled_os",
                        "lr_ged_sb_tsum_24",
                        "lr_splag_1_ged_sb",
                        "lr_splag_1_ged_os",
                        "lr_splag_1_ged_ns",
                        # Large-scale economic data with extreme skew
                        "lr_wdi_ny_gdp_mktp_kd",
                        "lr_wdi_nv_agr_totl_kn",
                        "lr_wdi_sm_pop_netm",
                        "lr_wdi_sm_pop_refg_or",
                    ],
                    # Bounded percentages and rates (0-100 scale)
                    "MinMaxScaler": [
                        "lr_wdi_sl_tlf_totl_fe_zs",
                        "lr_wdi_se_enr_prim_fm_zs",
                        "lr_wdi_sp_urb_totl_in_zs",
                        "lr_wdi_sh_sta_maln_zs",
                        "lr_wdi_sh_sta_stnt_zs",
                        "lr_wdi_dt_oda_odat_pc_zs",
                        "lr_wdi_ms_mil_xpnd_gd_zs",
                        # V-Dem indices (already 0-1 bounded)
                        "lr_vdem_v2x_horacc",
                        "lr_vdem_v2xnp_client",
                        "lr_vdem_v2x_veracc",
                        "lr_vdem_v2x_divparctrl",
                        "lr_vdem_v2xpe_exlpol",
                        "lr_vdem_v2x_diagacc",
                        "lr_vdem_v2xpe_exlgeo",
                        "lr_vdem_v2xpe_exlgender",
                        "lr_vdem_v2xpe_exlsocgr",
                        "lr_vdem_v2x_ex_party",
                        "lr_vdem_v2x_genpp",
                        "lr_vdem_v2xeg_eqdr",
                        "lr_vdem_v2xcl_prpty",
                        "lr_vdem_v2xeg_eqprotec",
                        "lr_vdem_v2x_ex_military",
                        "lr_vdem_v2xcl_dmove",
                        "lr_vdem_v2x_clphy",
                        "lr_vdem_v2x_hosabort",
                        "lr_vdem_v2xnp_regcorr",
                        # Topic model proportions (0-1 bounded)
                        "lr_topic_ste_theta0",
                        "lr_topic_ste_theta1",
                        "lr_topic_ste_theta2",
                        "lr_topic_ste_theta3",
                        "lr_topic_ste_theta4",
                        "lr_topic_ste_theta5",
                        "lr_topic_ste_theta6",
                        "lr_topic_ste_theta7",
                        "lr_topic_ste_theta8",
                        "lr_topic_ste_theta9",
                        "lr_topic_ste_theta10",
                        "lr_topic_ste_theta11",
                        "lr_topic_ste_theta12",
                        "lr_topic_ste_theta13",
                        "lr_topic_ste_theta14",
                        "lr_topic_ste_theta0_stock_t1_splag",
                        "lr_topic_ste_theta1_stock_t1_splag",
                        "lr_topic_ste_theta2_stock_t1_splag",
                        "lr_topic_ste_theta3_stock_t1_splag",
                        "lr_topic_ste_theta4_stock_t1_splag",
                        "lr_topic_ste_theta5_stock_t1_splag",
                        "lr_topic_ste_theta6_stock_t1_splag",
                        "lr_topic_ste_theta7_stock_t1_splag",
                        "lr_topic_ste_theta8_stock_t1_splag",
                        "lr_topic_ste_theta9_stock_t1_splag",
                        "lr_topic_ste_theta10_stock_t1_splag",
                        "lr_topic_ste_theta11_stock_t1_splag",
                        "lr_topic_ste_theta12_stock_t1_splag",
                        "lr_topic_ste_theta13_stock_t1_splag",
                        "lr_topic_ste_theta14_stock_t1_splag",
                    ],
                    # Growth rates (can be negative, roughly normal)
                    "StandardScaler->MinMaxScaler": ["lr_wdi_sp_pop_grow"],
                    # Mortality rates (positive, moderate skew)
                    "SqrtTransform->MinMaxScaler": ["lr_wdi_sp_dyn_imrt_fe_in"],
                    # Token counts (moderate skew)
                    "RobustScaler->MinMaxScaler": [
                        "lr_topic_tokens_t1",
                        "lr_topic_tokens_t1_splag",
                    ],
                }
            ]
        },
        
        # ============== TFT ARCHITECTURE ==============
        # CRITICAL: hidden_size was 2-8, which causes flat lines!
        # TFT needs sufficient capacity for attention mechanism to learn patterns.
        # Rule: hidden_size / num_attention_heads >= 16 for meaningful attention
        "hidden_size": {"values": [64, 128]},  # FIXED: was [2,4,8] - WAY too small!
        "lstm_layers": {"values": [1, 2]},
        "num_attention_heads": {"values": [4]},  # 64/4=16, 128/4=32 dimensions per head
        
        # Regularization - moderate dropout for sparse data
        "dropout": {"values": [0.1, 0.2]},
        
        # Attention configuration
        "full_attention": {"values": [True, False]},
        "feed_forward": {"values": ["GatedResidualNetwork"]},  # Best for TFT interpretability
        "add_relative_index": {"values": [True]},  # Helps with temporal patterns
        "use_static_covariates": {"values": [True, False]},
        "norm_type": {"values": ["LayerNorm"]},
        "use_reversible_instance_norm": {"values": [False]},
        
        # ============== LOSS FUNCTION ==============
        "loss_function": {"values": ["WeightedPenaltyHuberLoss"]},
        
        # Loss parameters tuned for zero-inflated data
        # More balanced weights to avoid collapsing to zero predictions
        "zero_threshold": {
            "distribution": "uniform",
            "min": 0.01,
            "max": 0.1,  # FIXED: was negative (-0.3 to -0.1)  dwhich is wrong
        },
        "delta": {
            "distribution": "log_uniform_values",
            "min": 0.5,
            "max": 2.0,
        },
        "non_zero_weight": {
            "distribution": "uniform",
            "min": 2.0,
            "max": 5.0,  # REDUCED: too high causes instability
        },
        "false_positive_weight": {
            "distribution": "uniform",
            "min": 1.0,
            "max": 2.5,
        },
        "false_negative_weight": {
            "distribution": "uniform",
            "min": 2.0,
            "max": 4.0,  # REDUCED: was 4-10, too aggressive
        },
    }

    sweep_config["parameters"] = parameters
    return sweep_config
