def get_sweep_config():
    """
    TiDE Hyperparameter Sweep Configuration - AsymmetricQuantileLoss
    =================================================================
    
    Strategy: Quantile Regression with Asymmetric Error Penalties
    --------------------------------------------------------------
    Quantile regression naturally handles asymmetric error costs without
    distributional assumptions. This avoids NB loss instability (Inf overflow)
    while achieving similar asymmetry through the tau parameter.
    
    Why AsymmetricQuantileLoss:
    - No distributional assumptions → no overflow risk
    - Linear loss in tails → robust to outliers and extreme events
    - tau controls asymmetry directly: tau=0.75 → 3× penalty for underestimation
    - Simpler to tune: only 3 hyperparameters (tau, non_zero_weight, threshold)
    
    Asymmetry Mechanics:
    - tau = 0.5: Symmetric (equivalent to MAE)
    - tau = 0.7: 2.3× penalty for underestimation vs overestimation
    - tau = 0.75: 3× penalty (3:1 FN:FP ratio)
    - tau = 0.8: 4× penalty
    
    BCD Optimization:
    - BCD = ∛(MTD × MSLE × log(1+MSE))
    - Higher tau → catches more events → lower MTD
    - non_zero_weight → focuses on conflict periods → improves MSLE
    - AsinhTransform → bounded predictions → controls MSE explosion
    
    Note: AsinhTransform still used for bounded training space.
    Loss computed in asinh-space, predictions inverse-transformed for evaluation.
    """
    sweep_config = {
        "method": "bayes",
        "name": "smol_cat_tide_quantile_v1_bcd",
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
        # Quantile loss is more stable than NB - can use smaller batches safely
        # Smaller batches = more gradient noise but better event detection
        "batch_size": {"values": [64, 128, 256, 512]}, 
        "n_epochs": {"values": [200]},
        "early_stopping_patience": {"values": [30]},
        "early_stopping_min_delta": {"values": [0.0001]},
        "force_reset": {"values": [True]},
        
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        # LR range compatible with all batch sizes in transformed space:
        # - Small batch (64): lower end of range (2e-5) prevents instability
        # - Large batch (512): can use higher end (1.5e-4) for faster convergence
        # Asinh transform makes this range safer than with raw counts
        "lr": {
            "distribution": "log_uniform_values",
            "min": 2e-5, 
            "max": 1.5e-4,
        },
        "weight_decay": {"values": [1e-6]},
        
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["CosineAnnealingWarmRestarts"]},
        "lr_scheduler_T_0": {"values": [25]},
        "lr_scheduler_T_mult": {"values": [1]},
        "lr_scheduler_eta_min": {"values": [1e-6]},
        "gradient_clip_val": {"values": [0.5, 1.0]},
        
        # ==============================================================================
        # SCALING
        # ==============================================================================
        # AsinhTransform compresses heavy-tailed counts into bounded range
        # Loss computed in asinh space (stable), predictions inverse-transformed for metrics
        "feature_scaler": {"values": [None]},
        "target_scaler": {"values": ["AsinhTransform"]},
        
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
        "temporal_width_past": {"values": [12]},
        "temporal_width_future": {"values": [12]},
        "temporal_hidden_size_past": {"values": [128, 256]},
        "temporal_hidden_size_future": {"values": [128, 256]},
        "temporal_decoder_hidden": {"values": [256]},
        
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        "use_layer_norm": {"values": [True, False]},
        "dropout": {"values": [0.1, 0.15]},
        "use_static_covariates": {"values": [False, True]},
        "use_reversible_instance_norm": {"values": [True, False]},
        
        # ==============================================================================
        # LOSS FUNCTION: AsymmetricQuantileLoss
        # ==============================================================================
        # Quantile regression with asymmetric error penalties
        # No distributional assumptions → stable, no overflow risk
        # tau controls FN/FP asymmetry directly
        "loss_function": {"values": ["AsymmetricQuantileLoss"]},
        
        # tau (quantile level): Controls asymmetry between under/overestimation
        # - tau = 0.5: Symmetric MAE
        # - tau = 0.7: 2.3× penalty for underestimation (FN:FP = 2.3:1)
        # - tau = 0.75: 3× penalty (FN:FP = 3:1)
        # - tau = 0.8: 4× penalty (FN:FP = 4:1)
        # Range 0.65-0.80: Favors catching events without excessive overprediction
        # Formula: underestimate penalty = tau, overestimate penalty = 1-tau
        "tau": {
            "distribution": "uniform",
            "min": 0.65,
            "max": 0.85,
        },
        
        # non_zero_weight: Extra weight for samples where target > threshold
        # With ~95% zeros in conflict data, non-zero targets need amplification
        # Range 2-10: Strong emphasis on conflict periods for MSLE optimization
        # Higher values help model focus on rare events but may cause overprediction
        "non_zero_weight": {
            "distribution": "uniform",
            "min": 2.0,
            "max": 10.0,
        },
        
        # Conversion: asinh(1)≈0.88, asinh(3)≈1.82, asinh(6)≈2.49, asinh(10)≈3.0
        "zero_threshold": {"values": [0.88, 1.82, 2.49]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config