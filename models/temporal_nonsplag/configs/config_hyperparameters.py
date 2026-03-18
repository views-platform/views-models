
def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.
    This configuration is "operational" so modifying these settings will impact the model's behavior during the training.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for training the model, which determine the model's behavior during the training phase.
    """
    
    hyperparameters = {
        "steps": [*range(1, 36 + 1, 1)], # won't be used

        "log_targets": True,  
        "log_features": ["lr_ged_sb", "lr_ged_os", "lr_ged_ns", "lr_pop_totl", "lr_vdem_v2x_libdem"],
        "feature_scaler": "StandardScaler", 
        "target_scaler": "StandardScaler",  

        "temporal_disaggregation": {
            "lr_gdp_pcap": {
                "method": "denton-cholette",
                "conversion": "sum",
            },
            "lr_pop_totl": {
                "method": "denton-cholette",
                "conversion": "average"
            },
            "lr_vdem_v2x_libdem": {
                "method": "denton-cholette",
                "conversion": "average"
            },

        },
        
        "input_chunk_length": 72,
        "output_chunk_length": 36,
        "output_chunk_shift": 0,

        "n_epochs": 100,

        "early_stopping_patience": 20,
        "early_stopping_min_delta": 0.01,
        "gradient_clip_val": 0.1, 

        "loss_function": "WeightedPenaltyHuberLoss",
        "zero_threshold": 0.01,
        "non_zero_weight": 5.0,
        "false_positive_weight": 15.0,
        "false_negative_weight": 10.0,
        "delta": 0.5,

        "optimizer_cls": "Adam",
        "lr": 3e-4,
        "weight_decay": 1e-3,
        "lr_scheduler_factor": 0.1,
        "lr_scheduler_patience": 3,
        "lr_scheduler_min_lr": 1e-6,

        "random_state": 42,
        "batch_size": 32,
        "num_samples": 1,
        "mc_dropout": False,



    }
    return hyperparameters
