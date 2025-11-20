
def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.
    This configuration is "operational" so modifying these settings will impact the model's behavior during the training.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for training the model, which determine the model's behavior during the training phase.
    """
    
    hyperparameters = {
        "steps": [*range(1, 36 + 1, 1)],
        "num_samples": 1,
        "mc_dropout": True,
        "log_features": [
            "lr_ged_sb",
            "lr_ged_ns",
            "lr_ged_os",
            "lr_acled_sb",
            "lr_acled_os",
            "lr_ged_sb_tsum_24",
            "lr_splag_1_ged_sb",
            "lr_splag_1_ged_os",
            "lr_splag_1_ged_ns",
            "lr_wdi_sm_pop_netm",
            "lr_wdi_sm_pop_refg_or",
            "lr_wdi_sp_dyn_imrt_fe_in",
            "lr_wdi_ny_gdp_mktp_kd"
        ],

        "activation": "SELU",
        "batch_size": 96,
        "delta": 0.22133381066373317,
        "dropout": 0.3,
        "early_stopping_min_delta": 0.001,
        "early_stopping_patience": 10,
        "false_negative_weight": 3.623807764266056,
        "false_positive_weight": 1.5063826868695804,
        "feature_scaler": "RobustScaler",
        "force_reset": True,
        "gradient_clip_val": 0.7024096798561852,
        "hidden_dim": 512,
        "input_chunk_length": 36,
        "log_targets": True,
        "loss_function": "WeightedPenaltyHuberLoss",
        "lr": 0.00010957030303834952,
        "lr_scheduler_factor": 0.4239910905014046,
        "lr_scheduler_min_lr": 0.000001,
        "lr_scheduler_patience": 7,
        "n_epochs": 300,
        "n_rnn_layers": 4,
        "non_zero_weight": 4.343113650298346,
        "output_chunk_shift": 0,
        "rnn_type": "LSTM",
        "target_scaler": "RobustScaler",
        "use_reversible_instance_norm": True,
        "weight_decay": 0.0005961383772670618,
        "zero_threshold": 0.24200337483851492
    }


    return hyperparameters