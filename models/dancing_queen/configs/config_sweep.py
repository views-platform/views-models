import math

def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    This configuration is "operational" so modifying it will change the search strategy, parameter ranges, and other settings for hyperparameter tuning aimed at optimizing model performance.

    Returns:
    - sweep_config (dict): A dictionary containing the configuration for hyperparameter sweeps, defining the methods and parameter ranges used to search for optimal hyperparameters.
    """

    sweep_config = {
        'method': 'bayes',
        'name': 'dancing_queen_rnn',
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 5,  # RNNs need time to learn temporal dependencies
            'eta': 2
        },
    }

    metric = {
        'name': 'time_series_wise_msle_mean_sb',
        'goal': 'minimize'
    }
    sweep_config['metric'] = metric

    parameters_dict = {
        # Temporal Configuration
        'steps': {'values': [[*range(1, 36 + 1, 1)]]},
        
        # Input length: RNNs can handle longer sequences but risk vanishing gradients
        'input_chunk_length': {'values': [36, 48, 60, 72]},  # Removed 24 (too short)
        
        # Training Configuration
        'batch_size': {'values': [64, 128, 256]},  # RNNs benefit from moderate batch sizes
        'n_epochs': {'values': [300]},  # Consolidated to single value
        'early_stopping_patience': {'values': [5, 7]},  # Removed 2,3 (too impatient)
        
        # Learning rate: RNNs are sensitive, need careful tuning
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,  # More conservative minimum
            'max': 1e-3,  # Upper bound for stability
        },
        
        # Weight decay: Regularization for recurrent connections
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-3,  # Reduced max from 1e-2
        },
        
        # Scaling: Critical for zero-inflated fatality data
        'feature_scaler': {
            'values': ['MinMaxScaler']
        },
        'target_scaler': {
            'values': ['MinMaxScaler']  # LogTransform best for count data
        },
        
        # RNN Architecture Parameters - CORE HYPERPARAMETERS
        
        # RNN type: Different architectures for temporal patterns
        'rnn_type': {'values': ['LSTM', 'GRU']},  # Removed vanilla RNN (vanishing gradients)
        
        # Hidden dimension: Size of recurrent state
        # Too small = can't capture patterns, too large = overfitting
        'hidden_dim': {'values': [32, 64, 128, 256]},  # Removed 16 (too small for conflict data)
        
        # Number of recurrent layers: Depth of temporal processing
        'n_rnn_layers': {'values': [1, 2, 3]},  # Removed 4 (too deep, gradient issues)
        
        # Dropout: Critical for RNN regularization
        'dropout': {'values': [0.2, 0.3, 0.4, 0.5]},  # Removed 0.0, 0.1 (need regularization)
        
        # Gradient clipping: Essential for RNN stability
        'gradient_clip_val': {
            'distribution': 'uniform',
            'min': 0.5,
            'max': 1.5  # Tighter range, catalog uses 0.8
        },
        
        # Activation: Output layer activation
        'activation': {'values': ['ReLU', 'Tanh']},  # Removed SELU (less common for RNNs)
        
        # Reversible Instance Normalization
        'use_reversible_instance_norm': {'values': [True, False]},
        
        # Loss Function Configuration - Critical for zero-inflated data
        'loss_function': {'values': ['WeightedPenaltyHuberLoss']},  # Only best loss
        
        # Zero threshold: What counts as "zero" in fatality data
        'zero_threshold': {
            'distribution': 'log_uniform_values',
            'min': 0.0001,  # ~1-10 fatalities in typical scaled space
            'max': 0.01,    # ~100-250 fatalities in typical scaled space
        },
        
        # False positives: Predicting conflict when there is none
        'false_positive_weight': {
            'distribution': 'uniform',
            'min': 2.0,
            'max': 20.0,  # Increased from 15
        },
        
        # False negatives: Missing actual conflicts (CRITICAL)
        'false_negative_weight': {
            'distribution': 'uniform',
            'min': 5.0,
            'max': 30.0,  # Increased from 15
        },
        
        # Non-zero weight: General importance of conflict events
        'non_zero_weight': {
            'distribution': 'uniform',
            'min': 3.0,
            'max': 20.0,  # Increased from 15
        },
        
        # Huber delta: Transition point between L2 and L1 loss
        'delta': {
            'distribution': 'uniform',
            'min': 0.01,
            'max': 5.0,  # Increased from 1.0
        },
    }
    sweep_config['parameters'] = parameters_dict

    return sweep_config