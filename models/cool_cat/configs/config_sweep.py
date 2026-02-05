def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    Optimized for TiDEModel on zero-inflated conflict fatalities data at country-month level.
    
    FIXES for weight collapse issue:
    - Drastically reduced weight_decay (was causing weights to shrink to ~1e-34)
    - Improved target scaling for zero-inflated data
    - Better learning rate to weight_decay ratio
    - Reduced overall regularization
    
    Returns:
    - sweep_config (dict): Configuration for hyperparameter sweeps.
    """

    sweep_config = {
        'method': 'bayes',
        'name': 'cool_cat_tide_balanced_v5_mtd',
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 20,  # Higher for scarce signal - need time to find patterns
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
        'input_chunk_length': {'values': [36, 48]},
        'output_chunk_shift': {'values': [0]},
        'mc_dropout': {'values': [True]},
        'random_state': {'values': [67]},

        # ============== TRAINING BASICS ==============
        # Larger batch sizes help stabilize gradients for zero-inflated data
        "batch_size": {"values": [256, 512, 1024, 2048]},
        'n_epochs': {'values': [100]},  # More epochs since we reduced regularization
        'early_stopping_patience': {'values': [15, 20, 25]},  # More patience for scarce signal
        "early_stopping_min_delta": {"values": [0.00005, 0.0001]},  # Smaller for [0,1] loss scale
        'force_reset': {'values': [True]},

        # ============== OPTIMIZER / SCHEDULER ==============
        # CRITICAL FIX: Weight decay was WAY too high, causing weight collapse!
        # Rule of thumb: weight_decay should be 10-100x smaller than lr
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 5e-5,
            'max': 1e-3,
        },
        # 'weight_decay': {
        #     'distribution': 'log_uniform_values',  # Log scale for better exploration
        #     'min': 1e-7,   # DRASTICALLY REDUCED (was 5e-4, caused weight collapse!)
        #     'max': 1e-5,   # DRASTICALLY REDUCED (was 5e-3)
        # },
        'weight_decay': {'values': [0]},
        'lr_scheduler_factor': {'values': [0.5]},  # Fixed for stability
        'lr_scheduler_patience': {'values': [8]},  # Fixed - consistent plateau detection
        'lr_scheduler_min_lr': {'values': [1e-6]},  # Higher floor
        "gradient_clip_val": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1.5,
        },

        # ============== SCALING ==============
        # For zero-inflated conflict data:
        # - AsinhTransform handles zeros naturally (unlike log)
        # - StandardScaler preserves gradient magnitude better than MinMaxScaler
        # - MinMaxScaler compresses gradients too much, contributing to weight collapse
        'feature_scaler': {'values': [None]},
        # Target is lr_ged_sb - use same scaling as the feature for consistency
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
                    "lr_topic_tokens_t1", "lr_topic_tokens_t1_splag",
                    # Growth rates (can be negative, roughly normal)
                    "lr_wdi_sp_pop_grow",
                    # Mortality rates (positive, skewed)
                    "lr_wdi_sp_dyn_imrt_fe_in",
                ],
            }]
        },

        # ============== TiDE ARCHITECTURE ==============
        # Larger hidden sizes help prevent weight collapse
        # The skip connection will always learn, but we need the main network to also learn
        'num_encoder_layers': {'values': [1, 2, 3]},  # Simpler for scarce signal
        'num_decoder_layers': {'values': [1, 2]},
        'decoder_output_dim': {'values': [32, 64, 128]},
        'hidden_size': {'values': [64, 128, 192]},  # Moderate sizes - avoid overfitting with scarce signal
        
        # Temporal processing for country-month data
        'temporal_width_past': {'values': [4, 6, 8]},
        'temporal_width_future': {'values': [6, 8, 12]},
        'temporal_hidden_size_past': {'values': [32, 48, 64, 96, 128]},  # Larger
        'temporal_hidden_size_future': {'values': [48, 64, 96]},
        'temporal_decoder_hidden': {'values': [64, 128, 192, 256]},
        
        # Regularization - REDUCED since we lowered weight decay
        # Too much regularization (dropout + weight_decay + layer_norm) caused collapse
        'use_layer_norm': {'values': [True, False]},  # Keep layer norm, it helps with zero-inflated
        'dropout': {'values': [0.05, 0.1, 0.15]},  # LOW dropout - preserve neurons that learn rare patterns
        'use_static_covariates': {'values': [False, True]},  # Simpler first
        # Reversible instance norm - True for non-stationary conflict data
        'use_reversible_instance_norm': {'values': [True, False]},

        # ============== LOSS FUNCTION ==============
        # For zero-inflated data, we need a loss that:
        # 1. Doesn't let the model just predict zeros everywhere
        # 2. Provides strong gradients for non-zero cases
        # 3. Doesn't overwhelm with the zero cases
        'loss_function': {'values': ['WeightedPenaltyHuberLoss']},
        
        # Zero threshold - what counts as "zero" after scaling
        'zero_threshold': {
            'distribution': 'uniform',  # uniform is fine for this small range
            'min': 0.05,   # Safely above 0 noise
            'max': 0.18,   # Just above where 1 fatality lands (~0.11)
        },
        # Delta for Huber loss - tighter range for consistent gradient flow
        'delta': {
            'distribution': 'uniform',
            'min': 0.8,
            'max': 1.0,  # Full L2
        },
        
        # Non-zero weight - narrower range for stability
        'non_zero_weight': {
            'distribution': 'uniform',
            'min': 4.0,
            'max': 7.0,  # Narrower range prevents conflicting gradients
        },
        
        "false_positive_weight": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1.0,
        },
        
        # False negative weight - narrower range
        'false_negative_weight': {
            'distribution': 'uniform',
            'min': 2.0,
            'max': 5.0,  # Narrower - still emphasizes missing conflicts
        },
    }

    sweep_config['parameters'] = parameters
    return sweep_config