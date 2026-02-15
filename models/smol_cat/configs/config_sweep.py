def get_sweep_config():
    """
    TiDE Hyperparameter Sweep Configuration - With NegativeBinomialLoss Option
    ===========================================================================
    
    Loss Function Options:
    ----------------------
    1. AsinhWeightedPenaltyHuberLoss (proven stable with non_zero_weight=30)
       - target_scaler: "AsinhTransform"
       - Works in asinh-transformed space with magnitude-aware weighting
       
    2. NegativeBinomialLoss (NEW - for overdispersed count data)
       - target_scaler: null (raw counts) OR "AsinhTransform" with inverse_transform=True
       - Theoretically suited for zero-inflated count data
       - NB variance formula: Var = μ + αμ² (handles overdispersion naturally)
       - alpha parameter controls dispersion (higher = more variance)
    
    Scaling Strategy for NB:
    ------------------------
    Using RAW COUNTS (target_scaler: null):
    - NB is mathematically designed for count data
    - softplus(prediction) ensures μ > 0
    - Log-likelihood formulation naturally handles large counts via log terms
    - No inverse transform complexity
    
    Note: Features still use mixed scaling (asinh for counts, standard for rates, etc.)
    Only the TARGET is raw counts.
    
    If training is unstable with raw counts, fallback options:
    A) Lower learning rate (1e-5 to 1e-4)
    B) Stricter gradient clipping (0.5 - 1.0)
    C) Use AsinhTransform with inverse_transform=True as backup
    """
    sweep_config = {
        "method": "bayes",
        "name": "smol_cat_tide_nbin_v1",
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
        "batch_size": {"values": [128, 512, 1024]}, 
        "n_epochs": {"values": [200]},
        "early_stopping_patience": {"values": [30]},
        "early_stopping_min_delta": {"values": [0.0001]},
        "force_reset": {"values": [True]},
        
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        # Conservative LR for raw count space
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-5, 
            "max": 5e-4,
        },
        "weight_decay": {"values": [1e-6]},
        
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["CosineAnnealingWarmRestarts"]},
        "lr_scheduler_T_0": {"values": [25]},
        "lr_scheduler_T_mult": {"values": [1]},
        "lr_scheduler_eta_min": {"values": [1e-6]},
        "gradient_clip_val": {"values": [0.5, 1.0]},  # Stricter for raw counts
        
        # ==============================================================================
        # SCALING
        # ==============================================================================
        # RAW COUNTS for target - NB is designed for this
        "feature_scaler": {"values": [None]},
        "target_scaler": {"values": [None]},  # Raw counts!
        
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
        # LOSS FUNCTION: NegativeBinomialLoss
        # ==============================================================================
        # NB is designed for overdispersed count data - ideal for conflict fatalities
        # Naturally handles zero-inflation through its distributional properties
        "loss_function": {"values": ["NegativeBinomialLoss"]},
        
        # Dispersion parameter α: controls Var = μ + αμ²
        "alpha": {
            "distribution": "uniform",
            "min": 0.3,
            "max": 0.9,
        },
        
        # Threshold for zero classification (in RAW COUNT space)
        # <1 fatality = essentially zero
        "zero_threshold": {"values": [1.0, 3.0, 6.0]},  # Testing around 1 fatality threshold
        
        # FP weight: false alarm penalty (predicting conflict when none exists)
        # Lower = more tolerant of false alarms (encourages non-zero predictions)
        "false_positive_weight": {
            "distribution": "uniform",
            "min": 0.9,
            "max": 1.0,
        },
        
        # FN weight: missed conflict penalty
        # Higher = penalize missing actual conflicts more
        "false_negative_weight": {
            "distribution": "uniform",
            "min": 1.0,
            "max": 1.5,
        },
        
        # Whether to estimate α from batch variance (experimental)
        # False = use fixed α from sweep
        "learn_alpha": {"values": [False]},
        
        # No inverse transform needed - using raw counts directly
        "inverse_transform": {"values": [False]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config