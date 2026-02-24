def get_sweep_config():
    """
    TiDE Hyperparameter Sweep Configuration - MagnitudeAwareHuberLoss
    ==================================================================

    Strategy: Asinh-space regression with magnitude-aware asymmetric weighting
    ---------------------------------------------------------------------------
    
    Why this approach for 85% zero-inflated conflict data:
    
    1. AsinhTransform on target: Compresses [0, 15000] → [0, ~10]
       - Stabilizes optimization landscape (model predicts in bounded space)
       - Preserves zero/non-zero distinction (asinh(0) = 0)
       - Makes gradient magnitudes comparable across the count range
    
    2. MagnitudeAwareHuberLoss: Explicitly encodes operational priorities
       - FN penalty: Missing conflicts penalized proportional to magnitude
       - FP penalty: False alarms tolerated more than missed events
       - Huber delta: L2 for small errors (precision), L1 for large (stability)
       - Magnitude scaling: Linear scaling ensures high-fatality events 
         get proportionally more gradient signal without explosion
    
    3. Feature-level scaling via feature_scaler_map:
       - Each feature scaled according to its statistical properties
       - Target scaled independently via AsinhTransform
    
    Sweep Metric: BCD (Balanced Conflict Deviation)
    - Geometric mean of MTD, P_level, P_tail, P_shape
    - Directly rewards volume accuracy, tail capture, and temporal correlation
    - MagnitudeAwareHuber's components map to BCD components:
      * FP/FN weights → P_level (volume)
      * Magnitude scaling → P_tail (extreme events)
      * Huber shape → P_shape (timing via stable gradients)
    
    LR Strategy:
    - Asinh space → stable gradients → can use wider LR range than raw counts
    - CosineAnnealingWarmRestarts with T_mult=2: progressively longer refinement
    - Gradient clipping still used for safety but less critical than NB
    """
    sweep_config = {
        "method": "bayes",
        "name": "cool_cat_tide_mahub_v14_bcd",
        "early_terminate": {"type": "hyperband", "min_iter": 30, "eta": 2},
        "metric": {"name": "time_series_wise_bcd_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # ==============================================================================
        # TEMPORAL CONFIGURATION
        # ==============================================================================
        "steps": {"values": [[*range(1, 36 + 1)]]},
        # input_chunk_length: How many months of history the model sees.
        # 36 = 3 years captures annual seasonality + medium-term conflict dynamics.
        # 48 = 4 years gives more context but increases memory and may overfit
        # on slow-moving features. For CM-level with ~200 series, 36 is safer.
        "input_chunk_length": {"values": [36]},
        "output_chunk_shift": {"values": [0]},
        "random_state": {"values": [67]},
        "output_chunk_length": {"values": [36]},
        "optimizer_cls": {"values": ["Adam"]},
        "mc_dropout": {"values": [True]},
        # num_samples=1 for point forecast evaluation during sweep.
        # MC dropout uncertainty is evaluated post-sweep with num_samples=100+.
        "num_samples": {"values": [1]},
        "n_jobs": {"values": [-1]},
        # ==============================================================================
        # TRAINING
        # ==============================================================================
        # Batch size 64: With ~200 countries × ~400 months ÷ chunks, dataset is small.
        # Smaller batches (64) give noisier gradients that act as implicit regularization,
        # helping the model not overfit to the zero-majority. Batch 128 smooths too much
        # for a dataset this small — the model converges to the zero-mode faster.
        # Batch 32 is too noisy for stable Huber training.
        "batch_size": {"values": [64]},
        # 300 epochs with patience 40: Conflict signal is sparse, so the model needs
        # more epochs to extract it. Early stopping prevents overfitting, but patience
        # must be long enough to survive LR warm restarts (T_0=30 → first restart at
        # epoch 30, second at 90). With patience=30, a restart could trigger early stop
        # before the model recovers. Patience=40 survives one full restart cycle.
        "n_epochs": {"values": [300]},
        "early_stopping_patience": {"values": [40]},
        "early_stopping_min_delta": {"values": [0.0001]},
        "force_reset": {"values": [True]},
        # ==============================================================================
        # OPTIMIZER
        # ==============================================================================
        # LR range for asinh-space training:
        # - Asinh compresses range → gradients are moderate → wider LR range is safe
        # - Lower bound 5e-5: Below this, model underfits on sparse signal
        # - Upper bound 5e-4: Above this, FN magnitude weighting causes instability
        # - Log-uniform: Explores orders of magnitude efficiently
        "lr": {
            "distribution": "log_uniform_values",
            "min": 5e-5,
            "max": 5e-4,
        },
        # Weight decay 1e-5: Light L2 regularization.
        # Too high (1e-4+) penalizes the large weights needed to predict rare events.
        # Too low (1e-7) provides no regularization benefit.
        # Sweep between 1e-6 and 1e-5 to let Bayes find the right balance.
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-5,
        },
        # ==============================================================================
        # LR SCHEDULER
        # ==============================================================================
        "lr_scheduler_cls": {"values": ["CosineAnnealingWarmRestarts"]},
        # T_0=30: First cycle is 30 epochs. With T_mult=2, subsequent cycles are
        # 60, 120 epochs. This gives:
        #   - Epochs 0-30: Exploration phase (high LR sweeps through loss landscape)
        #   - Epochs 30-90: First refinement (60 epochs at progressively lower LR)
        #   - Epochs 90-210: Deep refinement (120 epochs, model fine-tunes on rare events)
        # Previous T_0=25, T_mult=1 gave 8 identical 25-epoch cycles — too short
        # for the model to ever deeply learn rare conflict patterns.
        "lr_scheduler_T_0": {"values": [30]},
        "lr_scheduler_T_mult": {"values": [2]},
        "lr_scheduler_eta_min": {"values": [1e-6]},
        # Gradient clipping: Safety net, not primary stability mechanism.
        # In asinh space, gradients are naturally bounded (~10 max target).
        # Clip at 1.0 to catch rare outlier batches without affecting normal training.
        "gradient_clip_val": {"values": [1.0]},
        # ==============================================================================
        # SCALING
        # ==============================================================================
        # Target: AsinhTransform compresses [0, 15000] → [0, ~10]
        # This is THE critical design choice. Benefits:
        # 1. Model's linear output layer can reach all target values
        # 2. Gradient magnitudes are comparable across the count range
        # 3. Zero remains zero (asinh(0) = 0), preserving the zero/non-zero boundary
        # 4. MagnitudeAwareHuberLoss operates in this compressed space
        #    where threshold=0.88 corresponds to ~1 fatality
        "feature_scaler": {"values": [None]},
        "target_scaler": {"values": ["AsinhTransform"]},
        "feature_scaler_map": {
            "values": [
                {
                    "AsinhTransform": [
                        "lr_wdi_sm_pop_refg_or",
                        "lr_wdi_ny_gdp_mktp_kd",
                        "lr_wdi_nv_agr_totl_kn",
                        "lr_splag_1_ged_sb",
                        "lr_splag_1_ged_ns",
                        "lr_splag_1_ged_os",
                        "lr_topic_tokens_t1",
                        "lr_topic_tokens_t2",
                        "lr_topic_tokens_t13",
                        "lr_topic_tokens_t1_splag",
                    ],
                    "StandardScaler": [
                        "lr_ged_sb_delta",
                        "lr_ged_ns_delta",
                        "lr_ged_os_delta",
                        "lr_wdi_sm_pop_netm",
                        "lr_wdi_dt_oda_odat_pc_zs",
                        "lr_wdi_sp_pop_grow",
                        "lr_wdi_ms_mil_xpnd_gd_zs",
                        "lr_wdi_sp_dyn_imrt_fe_in",
                        "lr_wdi_sh_sta_stnt_zs",
                        "lr_wdi_sh_sta_maln_zs",
                    ],
                    "MinMaxScaler": [
                        "month",
                        "lr_wdi_sl_tlf_totl_fe_zs",
                        "lr_wdi_se_enr_prim_fm_zs",
                        "lr_wdi_sp_urb_totl_in_zs",
                        "lr_vdem_v2x_horacc",
                        "lr_vdem_v2x_veracc",
                        "lr_vdem_v2x_diagacc",
                        "lr_vdem_v2xnp_client",
                        "lr_vdem_v2xnp_regcorr",
                        "lr_vdem_v2xpe_exlpol",
                        "lr_vdem_v2xpe_exlgeo",
                        "lr_vdem_v2xpe_exlgender",
                        "lr_vdem_v2xpe_exlsocgr",
                        "lr_vdem_v2x_divparctrl",
                        "lr_vdem_v2x_ex_party",
                        "lr_vdem_v2x_ex_military",
                        "lr_vdem_v2x_genpp",
                        "lr_vdem_v2xeg_eqdr",
                        "lr_vdem_v2xcl_prpty",
                        "lr_vdem_v2xeg_eqprotec",
                        "lr_vdem_v2xcl_dmove",
                        "lr_vdem_v2x_clphy",
                        "lr_topic_ste_theta0_stock_t1",
                        "lr_topic_ste_theta0_stock_t2",
                        "lr_topic_ste_theta0_stock_t13",
                        "lr_topic_ste_theta1_stock_t1",
                        "lr_topic_ste_theta1_stock_t2",
                        "lr_topic_ste_theta1_stock_t13",
                        "lr_topic_ste_theta2_stock_t1",
                        "lr_topic_ste_theta2_stock_t2",
                        "lr_topic_ste_theta2_stock_t13",
                        "lr_topic_ste_theta3_stock_t1",
                        "lr_topic_ste_theta3_stock_t2",
                        "lr_topic_ste_theta3_stock_t13",
                        "lr_topic_ste_theta4_stock_t1",
                        "lr_topic_ste_theta4_stock_t2",
                        "lr_topic_ste_theta4_stock_t13",
                        "lr_topic_ste_theta5_stock_t1",
                        "lr_topic_ste_theta5_stock_t2",
                        "lr_topic_ste_theta5_stock_t13",
                        "lr_topic_ste_theta6_stock_t1",
                        "lr_topic_ste_theta6_stock_t2",
                        "lr_topic_ste_theta6_stock_t13",
                        "lr_topic_ste_theta7_stock_t1",
                        "lr_topic_ste_theta7_stock_t2",
                        "lr_topic_ste_theta7_stock_t13",
                        "lr_topic_ste_theta8_stock_t1",
                        "lr_topic_ste_theta8_stock_t2",
                        "lr_topic_ste_theta8_stock_t13",
                        "lr_topic_ste_theta9_stock_t1",
                        "lr_topic_ste_theta9_stock_t2",
                        "lr_topic_ste_theta9_stock_t13",
                        "lr_topic_ste_theta10_stock_t1",
                        "lr_topic_ste_theta10_stock_t2",
                        "lr_topic_ste_theta10_stock_t13",
                        "lr_topic_ste_theta11_stock_t1",
                        "lr_topic_ste_theta11_stock_t2",
                        "lr_topic_ste_theta11_stock_t13",
                        "lr_topic_ste_theta12_stock_t1",
                        "lr_topic_ste_theta12_stock_t2",
                        "lr_topic_ste_theta12_stock_t13",
                        "lr_topic_ste_theta13_stock_t1",
                        "lr_topic_ste_theta13_stock_t2",
                        "lr_topic_ste_theta13_stock_t13",
                        "lr_topic_ste_theta14_stock_t1",
                        "lr_topic_ste_theta14_stock_t2",
                        "lr_topic_ste_theta14_stock_t13",
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
                }
            ]
        },
        # ==============================================================================
        # TiDE ARCHITECTURE
        # ==============================================================================
        # Encoder/decoder depth: 2 layers is sufficient for CM-level (~200 series).
        # Deeper networks need more data per series to avoid overfitting.
        "num_encoder_layers": {"values": [2]},
        "num_decoder_layers": {"values": [2]},
        "decoder_output_dim": {"values": [128]},
        # hidden_size: Core capacity of the model.
        # 256 gives enough capacity to learn conflict-specific patterns.
        # 512 risks overfitting with ~200 country series.
        # Sweep 128 vs 256 to find the capacity sweet spot.
        "hidden_size": {"values": [128, 256]},
        # temporal_width: Controls how much temporal information flows through
        # the temporal encoder/decoder. Higher values capture more temporal
        # patterns but increase parameter count.
        # For CM-level monthly data, 24-64 captures seasonal + conflict dynamics.
        # 128 is overkill for monthly resolution (no sub-monthly patterns to learn).
        "temporal_width_past": {"values": [24, 64]},
        "temporal_width_future": {"values": [24, 64]},
        "temporal_decoder_hidden": {"values": [256]},
        # ==============================================================================
        # REGULARIZATION
        # ==============================================================================
        "use_layer_norm": {"values": [True]},
        # Dropout: Critical for zero-inflated data.
        # 0.20-0.30: Forces the model to learn redundant representations,
        # preventing over-reliance on any single feature for rare events.
        # Below 0.15: Model memorizes training zeros, predicts flat.
        # Above 0.35: Too much noise, model can't converge on sparse signal.
        "dropout": {
            "distribution": "uniform",
            "min": 0.20,
            "max": 0.30,
        },
        "use_static_covariates": {"values": [True]},
        # RevIN=False: Critical. RevIN normalizes each series to zero mean/unit var
        # before the model sees it. For zero-inflated data, this transforms the
        # majority zeros into negative values, destroying the zero/non-zero boundary
        # that the loss function relies on. Also distorts the asinh scale.
        "use_reversible_instance_norm": {"values": [False]},
        # ==============================================================================
        # LOSS FUNCTION: MagnitudeAwareHuberLoss
        # ==============================================================================
        # Operates in asinh-space where:
        #   asinh(0) = 0.0
        #   asinh(1) ≈ 0.88  (1 fatality)
        #   asinh(5) ≈ 2.31  (5 fatalities)
        #   asinh(25) ≈ 3.91 (25 fatalities)
        #   asinh(100) ≈ 5.30 (100 fatalities)
        #   asinh(1000) ≈ 7.60 (1000 fatalities)
        #   asinh(15000) ≈ 10.31 (max observed)
        "loss_function": {"values": ["MagnitudeAwareHuberLoss"]},
        # zero_threshold: Boundary between "zero" and "conflict" in asinh space.
        # 0.88 = asinh(1), meaning ≥1 fatality is classified as conflict.
        # This is the decision boundary for FP/FN weight assignment.
        # Don't sweep this — the boundary should be principled, not data-mined.
        "zero_threshold": {"values": [0.88]},
        # delta: Huber L2→L1 transition point.
        # In asinh space [0, ~10]:
        #   delta=1.5: L2 (quadratic) for errors <1.5 asinh units (~4 fatalities),
        #              L1 (linear) for larger errors. This means:
        #     - Small prediction errors get precise quadratic correction
        #     - Large errors (e.g., predicting 0 when actual is 1000) get linear
        #       treatment, preventing gradient explosion
        #   delta=1.0: More aggressive L1 transition, more robust to outliers
        #   delta=2.0: Wider L2 zone, more precise but less stable
        "delta": {
            "distribution": "uniform",
            "min": 1.0,
            "max": 2.0,
        },
        # non_zero_weight: ADDED to base weight (1.0) for any non-zero target.
        # With 85% zeros, the model sees ~6x more zero samples per epoch.
        # non_zero_weight=5 → TP weight = 1+5 = 6.0, approximately balancing
        # the class ratio. Range [3, 8] lets Bayes find the optimal balance.
        # Below 3: Model still dominated by zeros, predicts flat.
        # Above 10: Over-correction, model over-predicts everywhere.
        "non_zero_weight": {
            "distribution": "uniform",
            "min": 3.0,
            "max": 8.0,
        },
        # false_positive_weight: ABSOLUTE weight for predicting conflict where none exists.
        # Values < 1.0 mean FP is penalized LESS than a true negative error.
        # This is intentional: in conflict forecasting, a false alarm (FP) is
        # less costly than missing a conflict (FN). Range [0.3, 1.0]:
        #   0.3: Very tolerant of false alarms → model explores more
        #   0.8: Moderate FP penalty → model is more conservative
        # Below 0.3: Model predicts conflict everywhere ("escalation artifact")
        # Above 1.0: Conflicts with FN/non_zero_weight, pushes model back to zeros
        "false_positive_weight": {
            "distribution": "uniform",
            "min": 0.3,
            "max": 1.0,
        },
        # false_negative_weight: ADDED on top of (1 + non_zero_weight) for FN.
        # FN = model predicts zero but actual has conflict.
        # Total FN weight = 1 + non_zero_weight + false_negative_weight
        # With non_zero_weight=5, fn_weight=10: FN total = 16.0
        # With power-law magnitude scaling (exp=0.5) at target=7.6 (1000 fat):
        #   mult = 1 + (7.6/0.88)^0.5 ≈ 3.94x
        #   Effective FN weight: 16 × 3.94 ≈ 63x vs TN weight of 1.0
        #
        # Range [5, 15]:
        #   5: Mild FN penalty, model balances precision/recall
        #   15: Strong FN penalty, model prioritizes not missing conflicts
        # Above 20: Even with sqrt scaling, effective weights become large
        # for batch_size=64, causing gradient variance issues.
        "false_negative_weight": {
            "distribution": "uniform",
            "min": 1.0,
            "max": 15.0,
        },
        # magnitude_exponent: Power-law exponent for magnitude scaling.
        # Controls how aggressively the loss differentiates small vs large events.
        # Also controls adaptive delta: effective_delta = delta / (1+ratio)^(exp/2)
        #
        # Exponent values and their max multiplier (at target=9.0, threshold=0.88):
        #   0.3: mult=2.82x, δ_eff=0.93  — gentle, very stable
        #   0.5: mult=4.20x, δ_eff=0.76  — balanced (recommended)
        #   0.7: mult=6.33x, δ_eff=0.63  — aggressive but bounded
        #   1.0: mult=11.2x, δ_eff=0.50  — original linear (high variance)
        #
        # Sweep range [0.3, 0.7] lets Bayes find the optimal trade-off between
        # magnitude differentiation (higher exp) and gradient stability (lower exp).
        "magnitude_exponent": {
            "distribution": "uniform",
            "min": 0.3,
            "max": 0.7,
        },
        # ==============================================================================
        # TEMPORAL ENCODINGS
        # ==============================================================================
        "add_encoders": {
            "values": [
                {
                    "position": {"past": ["relative"], "future": ["relative"]},
                }
            ]
        },
    }

    sweep_config["parameters"] = parameters
    return sweep_config