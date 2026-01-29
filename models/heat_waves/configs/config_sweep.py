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
    
    Parameter Importance Analysis (vs MSLE):
    - weight_decay: +0.7 → MUCH LOWER weight decay needed
    - lr_scheduler_patience: +0.7 → shorter patience (faster decay)
    - input_chunk_length: +0.3 → shorter context is better
    - non_zero_weight: -0.4 → HIGHER values improve MSLE
    - lr_scheduler_factor: -0.3 → more aggressive decay helps
    - hidden_size: -0.2 → larger hidden_size helps
    - lstm_layers: -0.2 → more LSTM layers help
    - false_negative_weight: -0.2 → higher FN weight helps (anti-underprediction)
    - delta: -0.2 → slightly higher delta helps TFT
    
    Anti-Underprediction Strategy:
    - HIGH false_negative_weight + HIGH non_zero_weight → pick up conflict patterns
    - LOW weight_decay → less regularization, let model learn signal
    - Larger hidden_size + more lstm_layers → more capacity for patterns

    Returns:
    - sweep_config (dict): Configuration for hyperparameter sweeps.
    """

    sweep_config = {
        "method": "bayes",
        "name": "tft_heat_waves_balanced_v1",
        "early_terminate": {"type": "hyperband", "min_iter": 12, "eta": 2},
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ============== TEMPORAL CONFIGURATION ==============
        # input_chunk_length: +0.3 importance → shorter context is better
        "steps": {"values": [[*range(1, 36 + 1)]]},
        "input_chunk_length": {"values": [36, 48]},  # Shorter (was 48-72)
        "output_chunk_shift": {"values": [0]},
        
        # ============== TRAINING BASICS ==============
        # early_stopping_patience: +0.2 → slightly lower is better
        "batch_size": {"values": [32, 64]},
        "n_epochs": {"values": [350]},
        "early_stopping_patience": {"values": [12, 15, 18]},  # Reduced (was 18-25)
        "early_stopping_min_delta": {"values": [0.001]},
        "force_reset": {"values": [True]},
        
        # ============== OPTIMIZER / SCHEDULER ==============
        # weight_decay: +0.7 importance → MUCH LOWER needed
        # lr_scheduler_patience: +0.7 → shorter patience (faster decay)
        # lr_scheduler_factor: -0.3 → more aggressive decay
        "lr": {
            "distribution": "log_uniform_values",
            "min": 5e-6,
            "max": 1e-4,  # Lower range
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-6,   # MUCH LOWER (was 1e-5)
            "max": 5e-5,   # MUCH LOWER (was 5e-4)
        },
        "lr_scheduler_factor": {
            "distribution": "uniform",
            "min": 0.05,   # More aggressive (was 0.1)
            "max": 0.2,    # More aggressive (was 0.3)
        },
        "lr_scheduler_patience": {"values": [2, 3]},  # Shorter (was 3-5)
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
        # hidden_size: -0.2 importance → larger is better
        # lstm_layers: -0.2 importance → more layers help
        "hidden_size": {"values": [192, 256, 320]},  # Larger (was 128-256)
        "lstm_layers": {"values": [2, 3, 4]},  # More layers (was 2-3)
        "num_attention_heads": {"values": [4, 8]},  # Try more heads with larger hidden_size
        
        # hidden_continuous_size: controls processing of continuous variables
        # Default 8 is too small for 50+ features - scale with hidden_size
        "hidden_continuous_size": {"values": [32, 64]},
        
        # Regularization
        "dropout": {"values": [0.15, 0.25, 0.3]},  # Slightly lower dropout
        
        # Attention configuration
        "full_attention": {"values": [True]},  # Full attention for better patterns
        # feed_forward: GLU variants from "GLU Variants Improve Transformer" paper
        # SwiGLU/GEGLU often outperform GRN for learning sharp patterns
        "feed_forward": {"values": ["GatedResidualNetwork", "SwiGLU", "GEGLU"]},
        "add_relative_index": {"values": [True]},  # Helps with temporal patterns
        # skip_interpolation: skips interpolation in VariableSelectionNetwork
        # Can increase training speed without hurting accuracy
        "skip_interpolation": {"values": [False, True]},
        "use_static_covariates": {"values": [True]},  # Fixed for country-level
        "norm_type": {"values": ["LayerNorm", "RMSNorm"]},  # RMSNorm can be more stable
        "use_reversible_instance_norm": {"values": [False]},
        
        # ============== LOSS FUNCTION ==============
        # non_zero_weight: -0.4 importance → HIGHER values improve MSLE significantly
        # false_negative_weight: -0.2 → higher helps avoid underprediction
        # delta: -0.2 → slightly higher delta helps TFT (more L2-like)
        "loss_function": {"values": ["WeightedPenaltyHuberLoss"]},
        
        "zero_threshold": {"values": [0.01]},
        
        # delta: -0.2 importance → higher values help TFT
        "delta": {
            "distribution": "uniform",
            "min": 0.3,
            "max": 0.7,  # Higher range (was 0.1-0.4)
        },
        
        # non_zero_weight: -0.4 importance → MUCH HIGHER
        "non_zero_weight": {
            "distribution": "uniform",
            "min": 8.0,   # Higher (was 5.0)
            "max": 15.0,  # Higher (was 10.0)
        },
        
        # false_positive_weight: -0.009 → near zero importance, keep moderate
        "false_positive_weight": {
            "distribution": "uniform",
            "min": 1.5,
            "max": 3.0,
        },
        
        # false_negative_weight: -0.2 → higher helps avoid underprediction
        "false_negative_weight": {
            "distribution": "uniform",
            "min": 6.0,
            "max": 14.0,  # Higher to push predictions UP
        },
    }

    sweep_config["parameters"] = parameters
    return sweep_config
