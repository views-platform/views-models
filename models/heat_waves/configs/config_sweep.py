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
    
    Anti-Smoothing / Maximize y_hat Strategy:
    - HIGH false_negative_weight: penalize under-prediction heavily
    - LOW false_positive_weight: don't punish over-prediction
    - HIGH early_stopping_patience: let model train longer
    - Lower LR: more stable convergence
    - Lower delta: more L1-like loss (less mean regression)
    - Higher dropout: regularization helps avoid local minima

    Returns:
    - sweep_config (dict): Configuration for hyperparameter sweeps.
    """

    sweep_config = {
        "method": "bayes",
        "name": "tft_heat_waves_aggressive",
        "early_terminate": {"type": "hyperband", "min_iter": 10, "eta": 2},
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ============== TEMPORAL CONFIGURATION ==============
        "steps": {"values": [[*range(1, 36 + 1)]]},
        "input_chunk_length": {"values": [48, 60, 72]},  # Longer context helps avoid smoothing
        "output_chunk_shift": {"values": [0]},
        
        # ============== TRAINING BASICS ==============
        # Smaller batches help with sparse/zero-inflated data
        "batch_size": {"values": [32, 64]},
        "n_epochs": {"values": [400]},  # More epochs
        "early_stopping_patience": {"values": [18, 22, 25]},  # MUCH HIGHER
        "early_stopping_min_delta": {"values": [0.0005]},  # Tighter threshold
        "force_reset": {"values": [True]},
        
        # ============== OPTIMIZER / SCHEDULER ==============
        # TFT needs LOW learning rates to prevent attention collapse
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 2e-4,  # Lower upper bound
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 5e-4,
        },
        # More aggressive LR decay helps with stable convergence
        "lr_scheduler_factor": {
            "distribution": "uniform",
            "min": 0.1,
            "max": 0.3,  # More aggressive decay
        },
        "lr_scheduler_patience": {"values": [3, 4, 5]},  # Faster decay trigger
        "lr_scheduler_min_lr": {"values": [1e-7]},
        
        # CRITICAL: Gradient clipping prevents attention explosion -> flat lines
        "gradient_clip_val": {
            "distribution": "uniform",
            "min": 0.3,
            "max": 0.8,
        },
        # Scaling and transformation
        "feature_scaler": {"values": [None]},
        "target_scaler": {"values": ["AsinhTransform->MinMaxScaler"]},  # Fixed best option
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
        "hidden_size": {"values": [128, 256]},  # Larger for more capacity
        "lstm_layers": {"values": [2, 3]},  # Slightly deeper
        "num_attention_heads": {"values": [4]},  # 128/4=32, 256/4=64 dimensions per head
        
        # Regularization - higher dropout helps avoid smoothing
        "dropout": {"values": [0.2, 0.3, 0.35]},  # Higher dropout
        
        # Attention configuration
        "full_attention": {"values": [True]},  # Full attention for better patterns
        "feed_forward": {"values": ["GatedResidualNetwork"]},  # Best for TFT interpretability
        "add_relative_index": {"values": [True]},  # Helps with temporal patterns
        "use_static_covariates": {"values": [True]},  # Fixed for country-level
        "norm_type": {"values": ["LayerNorm"]},
        "use_reversible_instance_norm": {"values": [False]},
        
        # ============== LOSS FUNCTION ==============
        # High FN weight + Low FP weight → pushes predictions UP
        "loss_function": {"values": ["WeightedPenaltyHuberLoss"]},
        
        # Loss parameters tuned to maximize y_hat and avoid smoothing
        "zero_threshold": {"values": [0.01]},  # Fixed (data-dependent)
        
        # Lower delta = more L1-like, less regression to mean
        "delta": {
            "distribution": "uniform",
            "min": 0.1,
            "max": 0.4,  # Lower range
        },
        
        # Higher non_zero_weight to focus on predicting events
        "non_zero_weight": {
            "distribution": "uniform",
            "min": 5.0,
            "max": 10.0,
        },
        
        # LOW FP weight: Don't punish over-prediction
        "false_positive_weight": {
            "distribution": "uniform",
            "min": 1.0,
            "max": 2.0,  # Lower range
        },
        
        # HIGH FN weight: Heavily penalize under-prediction
        "false_negative_weight": {
            "distribution": "uniform",
            "min": 5.0,
            "max": 12.0,  # MUCH HIGHER - pushes predictions UP
        },
    }

    sweep_config["parameters"] = parameters
    return sweep_config
