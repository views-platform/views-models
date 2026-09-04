
def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.
    This configuration is "operational" so modifying these settings will impact the model's behavior during the training.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for training the model, which determine the model's behavior during the training phase.
    """
    
    hyperparameters = {
        'steps': [*range(1, 36 + 1, 1)],
        "temporal_disaggregation": {
            "lr_gdp_pcap": {
                "method": "denton-cholette",
                "conversion": "average",
            }
        },
        'window_months': 36,
        'output_chunk_length': 36,
    }
    return hyperparameters
