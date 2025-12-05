
def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.
    This configuration is "operational" so modifying these settings will impact the model's behavior during the training.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for training the model, which determine the model's behavior during the training phase.
    
    Feature Scaler Map:
    --------------------
    The `feature_scaler_map` parameter allows applying different scalers to different
    feature groups based on their data characteristics:
    
    - **UCDP/ACLED conflict data**: Zero-inflated, extreme outliers → RobustScaler or AsinhTransform
    - **WDI indicators**: Mixed scales, moderate outliers → RobustScaler or StandardScaler  
    - **V-Dem indices**: Bounded 0-1, well-behaved → MinMaxScaler or StandardScaler
    
    Two configuration formats are supported:
    
    1. Simple format (scaler → features):
        ```python
        "feature_scaler_map": {
            "RobustScaler": ["ged_sb", "ged_ns", "acled_sb"],
            "StandardScaler": ["wdi_ny_gdp_mktp_kd"],
            "MinMaxScaler": ["vdem_v2x_polyarchy"],
        }
        ```
    
    2. Named group format (group_name → {scaler, features}):
        ```python
        "feature_scaler_map": {
            "conflict": {
                "scaler": "RobustScaler",
                "features": ["ged_sb", "ged_ns", "acled_sb"]
            },
            "wdi": {
                "scaler": "StandardScaler",
                "features": ["wdi_ny_gdp_mktp_kd"]
            }
        }
        ```
    
    If `feature_scaler_map` is provided, it takes precedence over `feature_scaler`.
    Features not listed in the map will use `feature_scaler` as the default.
    """
    hyperparameters = {
        "steps": [*range(1, 36 + 1, 1)],

        "activation": "LeakyReLU",
        "batch_size": 16,
        "delta": 0.1213262276752683,
        "dropout": 0.3,
        "early_stopping_min_delta": 0.005,
        "early_stopping_patience": 10,
        "false_negative_weight": 2.5437452542589307,
        "false_positive_weight": 1.934082148492775,
        
        # Default feature scaler (used for features not in feature_scaler_map)
        "feature_scaler": "RobustScaler",
        
        # Feature-specific scalers: apply different scalers to different data sources
        # Uncomment to enable feature-specific scaling:
        # "feature_scaler_map": {
        #     "conflict": {
        #         "scaler": "RobustScaler",  # Best for zero-inflated conflict counts
        #         "features": ["ged_sb", "ged_ns", "ged_os", "acled_sb", "acled_os",
        #                      "ged_sb_tsum_24", "splag_1_ged_sb", "splag_1_ged_os", "splag_1_ged_ns"]
        #     },
        #     "wdi": {
        #         "scaler": "StandardScaler",  # Good for mixed-scale WDI indicators
        #         "features": ["wdi_sm_pop_netm", "wdi_sm_pop_refg_or", 
        #                      "wdi_sp_dyn_imrt_fe_in", "wdi_ny_gdp_mktp_kd"]
        #     },
        #     # V-Dem features would use the default RobustScaler
        # },
        
        "force_reset": True,
        "generic_architecture": False,
        "gradient_clip_val": 0.7257384795700205,
        "input_chunk_length": 12,
        "layer_widths": 32,
        "log_targets": True,
        "loss_function": "WeightedPenaltyHuberLoss",
        "lr": 0.00000112191691465149,
        "lr_scheduler_factor": 0.43147268008513817,
        "lr_scheduler_min_lr": 0.000001,
        "lr_scheduler_patience": 5,
        "n_epochs": 300,
        "non_zero_weight": 2.751958312084815,
        "num_blocks": 1,
        "num_layers": 1,
        "num_stacks": 1,
        "output_chunk_shift": 1,
        "target_scaler": "AsinhTransform",  # Best for zero-inflated targets
        "weight_decay": 0.00028874387372725287,
        "zero_threshold": 0.2642766420812154,

        "log_features": ["lr_ged_sb", "lr_ged_ns", "lr_ged_os", "lr_acled_sb", "lr_acled_os", 
                 "lr_ged_sb_tsum_24", "lr_splag_1_ged_sb", "lr_splag_1_ged_os", "lr_splag_1_ged_ns", 
                 "lr_wdi_sm_pop_netm", "lr_wdi_sm_pop_refg_or", "lr_wdi_sp_dyn_imrt_fe_in", "lr_wdi_ny_gdp_mktp_kd",
                 ],

        "num_samples": 1,
        "mc_dropout": True,
    }
    return hyperparameters

