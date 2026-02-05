def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    Optimized for TFT on zero-inflated, right-skewed conflict data at country-month level.
    
    GRADIENT FLOW FIXES:
    - Changed MinMaxScaler -> StandardScaler (preserves gradient magnitude)
    - Increased delta (0.5-2.0) for stronger L2-like gradients
    - Removed weight_decay (was suppressing learning of rare patterns)
    - Reduced dropout for scarce signal
    - Increased patience for early stopping and hyperband

    TFT Architecture Notes:
    - hidden_size must be >= 32 (ideally 64-128) for meaningful attention
    - hidden_size must be divisible by num_attention_heads with quotient >= 16
    - Gradient clipping prevents attention collapse
    - add_relative_index helps capture temporal patterns

    Returns:
    - sweep_config (dict): Configuration for hyperparameter sweeps.
    """

    sweep_config = {
        "method": "bayes",
        "name": "tft_heat_waves_idek_anymore_v2_mtd",
        "early_terminate": {"type": "hyperband", "min_iter": 20, "eta": 2},  # Higher for scarce signal
        "metric": {"name": "time_series_wise_mtd_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ============== TEMPORAL CONFIGURATION ==============
        "steps": {"values": [[*range(1, 36 + 1)]]},
        "input_chunk_length": {"values": [24, 36, 48]},
        "output_chunk_shift": {"values": [0]},
        "output_chunk_length": {"values": [36]},
        "random_state": {"values": [42]},
        "mc_dropout": {"values": [True]},
        
        # ============== TRAINING BASICS ==============
        "batch_size": {"values": [32, 64, 128, 256]},  # Moderate sizes
        "n_epochs": {"values": [150]},  # More epochs with better hyperparams
        "early_stopping_patience": {"values": [15, 20, 25]},  # MORE patience for scarce signal
        "early_stopping_min_delta": {"values": [0.0001, 0.0005]},  # Smaller delta
        "force_reset": {"values": [True]},
        
        # ============== OPTIMIZER / SCHEDULER ==============
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 5e-4,
        },
        # CRITICAL: No weight decay for scarce signal
        "weight_decay": {"values": [0]},
        "lr_scheduler_factor": {"values": [0.5]},  # Fixed for stability
        "lr_scheduler_patience": {"values": [8]},  # Fixed - consistent plateau detection
        "lr_scheduler_min_lr": {"values": [1e-6]},
        
        # Gradient clipping - tight range for consistent training
        "gradient_clip_val": {
            "distribution": "uniform",
            "min": 0.8,
            "max": 1.2,
        },
        
        # ============== SCALING ==============
        # CRITICAL: StandardScaler instead of MinMaxScaler for gradient preservation
        "feature_scaler": {"values": [None]},
        'target_scaler': {'values': ['AsinhTransform->MinMaxScaler']},
        'feature_scaler_map': {
            'values': [{
                # Zero-inflated conflict counts - Asinh + StandardScaler preserves gradients
                "AsinhTransform->MinMaxScaler": [
                    "lr_ged_sb", "lr_ged_ns", "lr_ged_os",
                    "lr_acled_sb", "lr_acled_sb_count", "lr_acled_os",
                    "lr_ged_sb_tsum_24",
                    "lr_splag_1_ged_sb", "lr_splag_1_ged_os", "lr_splag_1_ged_ns",
                    # Large-scale economic data with extreme skew
                    "lr_wdi_ny_gdp_mktp_kd", "lr_wdi_nv_agr_totl_kn",
                    "lr_wdi_sm_pop_netm", "lr_wdi_sm_pop_refg_or",
                    # Mortality rates (positive, skewed)
                    "lr_wdi_sp_dyn_imrt_fe_in"
                ],
                # Bounded percentages, V-Dem indices, and growth rates - StandardScaler works fine
                "MinMaxScaler": [
                    "lr_wdi_sl_tlf_totl_fe_zs", "lr_wdi_se_enr_prim_fm_zs",
                    "lr_wdi_sp_urb_totl_in_zs", "lr_wdi_sh_sta_maln_zs", "lr_wdi_sh_sta_stnt_zs",
                    "lr_wdi_dt_oda_odat_pc_zs", "lr_wdi_ms_mil_xpnd_gd_zs",
                    # V-Dem indices (already 0-1 bounded)
                    "lr_vdem_v2x_horacc", "lr_vdem_v2xnp_client", "lr_vdem_v2x_veracc",
                    "lr_vdem_v2x_divparctrl", "lr_vdem_v2xpe_exlpol", "lr_vdem_v2x_diagacc",
                    "lr_vdem_v2xpe_exlgeo", "lr_vdem_v2xpe_exlgender", "lr_vdem_v2xpe_exlsocgr",
                    "lr_vdem_v2x_ex_party", "lr_vdem_v2x_genpp", "lr_vdem_v2xeg_eqdr",
                    "lr_vdem_v2xcl_prpty", "lr_vdem_v2xeg_eqprotec", "lr_vdem_v2x_ex_military",
                    "lr_vdem_v2xcl_dmove", "lr_vdem_v2x_clphy", "lr_vdem_v2x_hosabort",
                    "lr_vdem_v2xnp_regcorr",
                    # Topic model proportions (0-1 bounded)
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
                    # Growth rates (can be negative, roughly normal)
                    "lr_wdi_sp_pop_grow",
                    "lr_topic_tokens_t1", "lr_topic_tokens_t1_splag"
                ],
            }]
        },
        
        # ============== TFT ARCHITECTURE ==============
        "hidden_size": {"values": [64, 96, 128]},
        "lstm_layers": {"values": [2, 3]},  # Simpler for scarce signal
        "num_attention_heads": {"values": [2, 4, 8]},
        
        # hidden_continuous_size: scale with number of features (50+)
        "hidden_continuous_size": {"values": [16, 32, 48]},  # LARGER (was 8-24)
        
        # Regularization - LOW for scarce signal
        "dropout": {"values": [0.05, 0.1, 0.15]},  # MUCH LOWER (was 0.2-0.4)
        
        # Attention configuration
        "full_attention": {"values": [True]},
        "feed_forward": {"values": ["GatedResidualNetwork", "SwiGLU", "GEGLU"]},  # Best performers
        "add_relative_index": {"values": [True]},
        "skip_interpolation": {"values": [False]},  # Keep interpolation for better learning
        "use_static_covariates": {"values": [False]},  # Simpler first
        "norm_type": {"values": ["RMSNorm", "LayerNorm"]},
        "use_reversible_instance_norm": {"values": [True, False]},  # Already using AsinhTransform
        
        # ============== LOSS FUNCTION ==============
        "loss_function": {"values": ["WeightedPenaltyHuberLoss"]},
        
        "zero_threshold": {
            "distribution": "log_uniform_values",
            "min": 0.01,
            "max": 0.2,
        },
        
        # CRITICAL: Delta - tight range for consistent gradient flow
        "delta": {
            "distribution": "uniform",
            "min": 0.8,
            "max": 1.5,
        },
        
        # Non-zero weight - narrower range for stability
        "non_zero_weight": {
            "distribution": "uniform",
            "min": 4.0,
            "max": 7.0,  # Narrower range prevents conflicting gradients
        },
        
        "false_positive_weight": {"values": [1.0]},  # Fixed - variable weighting causes instability
        
        # False negative weight - narrower range
        "false_negative_weight": {
            "distribution": "uniform",
            "min": 2.0,
            "max": 5.0,  # Narrower - still emphasizes missing conflicts
        },
    }
    sweep_config["parameters"] = parameters
    return sweep_config