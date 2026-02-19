def get_sweep_config():
    """
    TiDE Hyperparameter Sweep Configuration - NegativeBinomialLoss
    ===============================================================
    
    Strategy: Probabilistic Count Modeling with Overdispersion
    -----------------------------------------------------------
    Uses Negative Binomial distribution for zero-inflated, overdispersed count data.
    Model predicts RAW COUNTS directly (no target transformation).
    
    Why NegativeBinomialLoss:
    - Proper count distribution: NB is designed for discrete non-negative data
    - Overdispersion: Var = μ + αμ² handles conflict data where Var >> Mean
    - Zero-inflation: NB naturally assigns high probability to zero
    - Calibrated uncertainty: Better tail coverage than Gaussian losses
    
    Parameterization:
    - μ (mean): predicted via softplus(model_output), always > 0
    - α (dispersion): controls overdispersion, higher = more variance
    - P(Y=0 | μ, α) = (1 + αμ)^(-1/α)
    
    Alpha Interpretation:
    - α → 0: approaches Poisson (Var = Mean)
    - α = 0.5-1.0: mild overdispersion (typical for low-intensity conflict)
    - α = 1.0-2.0: moderate overdispersion (typical for high-intensity regions)
    - α > 2.0: severe overdispersion (extreme conflict spikes)
    
    Example with α=1.0, μ=10:
    - Variance = 10 + 1×10² = 110 (vs Poisson Var=10)
    - P(Y=0) = (1 + 10)^(-1) = 0.091
    
    FN/FP Weighting:
    - false_negative_weight > 1.0: Missing conflict is worse
    - false_positive_weight < 1.0: False alarms less costly
    
    LR Considerations:
    - Raw counts → larger gradients than transformed data
    - Use lower LR range: 1e-5 to 2e-4
    - Combined with gradient clipping for stability
    """
    sweep_config = {
        "method": "bayes",
        "name": "cool_cat_tide_nbinomial_v5_bcd",
        "early_terminate": {"type": "hyperband", "min_iter": 30, "eta": 2},
        "metric": {"name": "time_series_wise_bcd_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        "input_chunk_length": {"values": [36, 48]},
        "output_chunk_shift": {"values": [0]},
        "random_state": {"values": [67]},
        "output_chunk_length": {"values": [36]},
        "optimizer_cls": {"values": ["Adam"]},
        "mc_dropout": {"values": [True]},
        "num_samples": {"values": [1]},
        "n_jobs": {"values": [-1]},
        
        # ==============================================================================
        # TRAINING
        # ==============================================================================
        # With raw counts, larger batches stabilize gradient estimates
        # Batch 64-128 recommended for NB loss stability
        "batch_size": {"values": [64, 128]}, 
        "n_epochs": {"values": [200]},
        "early_stopping_patience": {"values": [30]},
        "early_stopping_min_delta": {"values": [0.0001]},
        "force_reset": {"values": [True]},
        
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        # LR range for NegativeBinomialLoss with raw counts:
        # - Raw counts → larger gradients → need lower LR
        # - NB has log-gamma computations → sensitive to instability
        # - Batch 64-128 with gradient clipping enables slightly higher LR
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 2e-4,
        },
        "weight_decay": {"values": [1e-6]},
        
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["CosineAnnealingWarmRestarts"]},
        "lr_scheduler_T_0": {"values": [25]},
        "lr_scheduler_T_mult": {"values": [1]},
        "lr_scheduler_eta_min": {"values": [1e-6]},
        # Gradient clipping critical for NB with raw counts (large gradients)
        "gradient_clip_val": {"values": [0.5, 1.0]},
        
        # ==============================================================================
        # SCALING
        # ==============================================================================
        # NegativeBinomialLoss requires RAW COUNTS for proper NB semantics
        # Model predicts raw counts through softplus (ensures μ > 0)
        # Input features still scaled for stable forward pass
        "feature_scaler": {"values": [None]},
        "target_scaler": {"values": [None]},
        
        "feature_scaler_map": {
            "values": [
                {
                    # Conflict history: Asinh transform for heavy tails
                    "AsinhTransform": [
                        "lr_acled_sb", "lr_acled_os",
                        "lr_wdi_sm_pop_refg_or",
                        "lr_wdi_ny_gdp_mktp_kd", "lr_wdi_nv_agr_totl_kn",
                        "lr_splag_1_ged_sb", "lr_splag_1_ged_ns", "lr_splag_1_ged_os",
                    ],
                    # Indices/Rates: Standard scaling
                    "StandardScaler": [
                        "lr_wdi_sm_pop_netm", "lr_wdi_dt_oda_odat_pc_zs",
                        "lr_wdi_sp_pop_grow", "lr_wdi_ms_mil_xpnd_gd_zs",
                        "lr_wdi_sp_dyn_imrt_fe_in", "lr_wdi_sh_sta_stnt_zs",
                        "lr_wdi_sh_sta_maln_zs",
                    ],
                    # Bounded [0,1] features
                    "MinMaxScaler": [
                        "lr_wdi_sl_tlf_totl_fe_zs", "lr_wdi_se_enr_prim_fm_zs",
                        "lr_wdi_sp_urb_totl_in_zs",
                        "lr_vdem_v2x_horacc", "lr_vdem_v2xnp_client", "lr_vdem_v2x_veracc",
                        "lr_vdem_v2x_divparctrl", "lr_vdem_v2xpe_exlpol", "lr_vdem_v2x_diagacc",
                        "lr_vdem_v2xpe_exlgeo", "lr_vdem_v2xpe_exlgender", "lr_vdem_v2xpe_exlsocgr",
                        "lr_vdem_v2x_ex_party", "lr_vdem_v2x_genpp", "lr_vdem_v2xeg_eqdr",
                        "lr_vdem_v2xcl_prpty", "lr_vdem_v2xeg_eqprotec", "lr_vdem_v2x_ex_military",
                        "lr_vdem_v2xcl_dmove", "lr_vdem_v2x_clphy", "lr_vdem_v2xnp_regcorr",
                        # Topics
                        "lr_topic_ste_theta0", "lr_topic_ste_theta1", "lr_topic_ste_theta2",
                        "lr_topic_ste_theta3", "lr_topic_ste_theta4", "lr_topic_ste_theta5",
                        "lr_topic_ste_theta6", "lr_topic_ste_theta7", "lr_topic_ste_theta8",
                        "lr_topic_ste_theta9", "lr_topic_ste_theta10", "lr_topic_ste_theta11",
                        "lr_topic_ste_theta12", "lr_topic_ste_theta13", "lr_topic_ste_theta14",
                        # Topic Lags
                        "lr_topic_ste_theta0_stock_t1_splag", "lr_topic_ste_theta1_stock_t1_splag",
                        "lr_topic_ste_theta2_stock_t1_splag", "lr_topic_ste_theta3_stock_t1_splag",
                        "lr_topic_ste_theta4_stock_t1_splag", "lr_topic_ste_theta5_stock_t1_splag",
                        "lr_topic_ste_theta6_stock_t1_splag", "lr_topic_ste_theta7_stock_t1_splag",
                        "lr_topic_ste_theta8_stock_t1_splag", "lr_topic_ste_theta9_stock_t1_splag",
                        "lr_topic_ste_theta10_stock_t1_splag", "lr_topic_ste_theta11_stock_t1_splag",
                        "lr_topic_ste_theta12_stock_t1_splag", "lr_topic_ste_theta13_stock_t1_splag",
                        "lr_topic_ste_theta14_stock_t1_splag",
                    ],
                }
            ]
        },
        
        # ==============================================================================
        # TiDE ARCHITECTURE
        # ==============================================================================
        "num_encoder_layers": {"values": [2]},
        "num_decoder_layers": {"values": [2]},
        "decoder_output_dim": {"values": [128]},
        "hidden_size": {"values": [128, 256]},
        "temporal_width_past": {"values": [24, 64, 128]},
        "temporal_width_future": {"values": [24, 64, 128]},
        "temporal_hidden_size_past": {"values": [128, 256]},
        "temporal_hidden_size_future": {"values": [128, 256]},
        "temporal_decoder_hidden": {"values": [256]},
        
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        "use_layer_norm": {"values": [True, False]},
        "dropout": {"values": [0.15, 0.25]},
        "use_static_covariates": {"values": [False, True]},
        "use_reversible_instance_norm": {"values": [True, False]},
        
        # ==============================================================================
        # LOSS FUNCTION: NegativeBinomialLoss
        # ==============================================================================
        # Negative Binomial for overdispersed count data
        # NLL: -log P(y | μ, α) with softplus(pred) → μ
        "loss_function": {"values": ["NegativeBinomialLoss"]},
        
        # alpha (dispersion): Controls overdispersion Var = μ + αμ²
        # - alpha = 0.5: mild overdispersion
        # - alpha = 1.0: moderate (Var = μ + μ²)
        # - alpha = 2.0: severe overdispersion
        # Higher alpha → more probability mass at extremes
        "alpha": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 2.0,
        },
        
        # false_negative_weight: Penalty multiplier for missing conflict
        # FN = model predicts low but actual is high
        # Range 1.5-3.0: moderate to strong preference for catching events
        "false_negative_weight": {
            "values": [1.0],
        },
        
        # false_positive_weight: Penalty multiplier for false alarms
        # FP = model predicts high but actual is zero/low
        # Range 0.5-1.0: equal or reduced penalty for false positives
        "false_positive_weight": {
            "distribution": "uniform",
            "min": 1.0,
            "max": 12.0,
        },
        
        # zero_threshold: Raw count threshold to distinguish zero from non-zero
        # For raw counts, 0.5 means < 1 fatality classified as zero
        "zero_threshold": {"values": [1, 3, 4]},
        
        # learn_alpha: Estimate dispersion from batch variance
        # If True, overrides fixed alpha with moment-based estimate
        "learn_alpha": {"values": [False]},
        
        # inverse_transform: Not needed since target_scaler is None
        "inverse_transform": {"values": [False]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config