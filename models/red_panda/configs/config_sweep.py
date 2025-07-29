def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    This configuration is "operational" so modifying it will change the search strategy, parameter ranges, and other settings for hyperparameter tuning aimed at optimizing model performance.

    Returns:
    - sweep_config (dict): A dictionary containing the configuration for hyperparameter sweeps, defining the methods and parameter ranges used to search for optimal hyperparameters.
    """

    sweep_config = {
        'method': 'bayes',  # Other options: 'grid', 'bayes'
        'name': 'tide_proto',
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 5,
            'eta': 2
        },
    }

    # Example metric setup:
    metric = {
        'name': 'time_series_wise_msle_mean_sb',
        'goal': 'minimize'
    }
    sweep_config['metric'] = metric

    # Example parameters setup:
    parameters_dict = {
        'steps': {'values': [[*range(1, 36 + 1, 1)]]},
        'input_chunk_length': {'values': [24, 36, 48, 60, 72]},
        'batch_size': {'values': [64, 128, 256]},
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-3,
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-2
        },
        'n_epochs': {'values': [2, 5, 10, 50, 75, 120]},
        'early_stopping_patience': {'values': [2, 3, 5, 7]},
        'feature_scaler': {
            'values': ['StandardScaler', 'RobustScaler', 'MinMaxScaler', 'MaxAbsScaler', 'YeoJohnsonTransform',  None]
        },
        'target_scaler': {
            'values': ['StandardScaler', 'RobustScaler', 'MinMaxScaler', 'MaxAbsScaler', 'YeoJohnsonTransform',  None]
        },
        # 'delta': {'values': [0.01, 0.05, 0.1, 0.2, 0.5]},
        'hidden_size': {'values': [128, 256, 512, 1024]},
        'num_encoder_layers': {'values': [1, 2, 3]},
        'num_decoder_layers': {'values': [1, 2, 3]},
        'dropout': {'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]},
        'gradient_clip_val': {'values': [0.2, 0.5, 0.8, 1.0]},
        'use_layer_norm': {'values': [True, False]},
        'decoder_output_dim': {'values': [8, 16, 32]},
        'temporal_decoder_hidden': {'values': [16, 32, 64]},
        # 'zero_threshold': {'values': [0.001, 0.005, 0.01, 0.05]},

        'loss_function': {
            'values': [
                'WeightedHuberLoss',
                # 'TimeAwareWeightedHuberLoss',
                'SpikeFocalLoss',
                # 'AsymmetricSpikeLoss',
                # 'LogSpaceLoss',
                # 'ZeroInflatedTweedieLoss',
                # 'HybridSpikeLoss'
            ]
        },
        
        # Common loss parameters
        'zero_threshold': {
            'distribution': 'uniform',
            'min': 0.001,
            'max': 0.1
        },
        'non_zero_weight': {
            'distribution': 'uniform',
            'min': 1.0,
            'max': 25.0,
        },
        'delta': {
            'distribution': 'uniform',
            'min': 0.01,
            'max': 1.0
        },
        
        # Loss-specific parameters
        # For WeightedSmoothL1Loss
        # 'beta': {
        #     'distribution': 'uniform',
        #     'min': 0.1,
        #     'max': 0.5
        # },
        # 'zero_weight': {
        #     'distribution': 'uniform',
        #     'min': 0.1,
        #     'max': 1.0
        # },
        
        # # For TimeAwareWeightedHuberLoss
        # 'decay_factor': {
        #     'distribution': 'uniform',
        #     'min': 0.8,
        #     'max': 0.99
        # },
        
        # For SpikeFocalLoss
        'focal_alpha': {
            'distribution': 'uniform',
            'min': 0.5,
            'max': 0.9
        },
        'focal_gamma': {
            'distribution': 'uniform',
            'min': 1.0,
            'max': 10.0,
        },
        
        # # For AsymmetricSpikeLoss
        # 'under_pred_penalty': {
        #     'distribution': 'q_log_uniform',
        #     'min': 1.0,
        #     'max': 10.0,
        #     'q': 1.0
        # },
        # 'over_pred_penalty': {
        #     'distribution': 'uniform',
        #     'min': 0.1,
        #     'max': 1.0
        # },
        
        # # For ZeroInflatedTweedieLoss
        # 'tweedie_p': {
        #     'distribution': 'uniform',
        #     'min': 1.1,
        #     'max': 1.9
        # },
        # 'tweedie_eps': {
        #     'distribution': 'log_uniform_values',
        #     'min': 1e-9,
        #     'max': 1e-5
        # },
        
        # # For HybridSpikeLoss
        # 'hybrid_alpha': {
        #     'distribution': 'uniform',
        #     'min': 0.4,
        #     'max': 0.8
        # },
        'spike_threshold': {
            'distribution': 'uniform',
            'min': 0.05,
            'max': 10,
        }

    }
    sweep_config['parameters'] = parameters_dict

    return sweep_config
