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
        'name': 'good_life_transformer',
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 5,  # Transformers need time for attention to converge
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
        
        # Input length: Transformers handle longer sequences well (quadratic attention)
        'input_chunk_length': {'values': [36, 48, 60, 72, 84]},  # Added 84
        
        # Training Configuration
        'batch_size': {'values': [64, 128, 256]},  # Transformers memory-intensive
        'n_epochs': {'values': [300]},  # Consolidated (let early stopping decide)
        'early_stopping_patience': {'values': [5, 7]},  # Removed impatient values
        
        # Learning rate: Transformers are sensitive, need careful tuning
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-3,  # Conservative upper bound
        },
        
        # Weight decay: Important for transformer regularization
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-3,  # Reduced from 1e-2
        },
        
        # Scaling: Critical for zero-inflated fatality data
        'feature_scaler': {
            'values': ['MaxAbsScaler', 'MinMaxScaler']
        },
        'target_scaler': {
            'values': ['LogTransform', 'MinMaxScaler', 'MaxAbsScaler']  # LogTransform best for count data
        },
        
        # Transformer Architecture Parameters - CORE HYPERPARAMETERS
        
        # Model dimension: Must be divisible by nhead
        # d_model = nhead * d_k (dimension per head)
        'd_model': {'values': [64, 128, 256, 512]},  # Removed 32 (too small for attention)
        
        # Number of attention heads: Must divide d_model evenly
        # More heads = more diverse attention patterns
        'nhead': {'values': [2, 4, 8]},  # Removed 16 (requires d_model>=256)
        
        # Encoder layers: Process input sequence
        'num_encoder_layers': {'values': [2, 3, 4]},  # Removed 1 (too shallow)
        
        # Decoder layers: Generate output sequence
        'num_decoder_layers': {'values': [2, 3, 4]},  # Removed 1 (too shallow)
        
        # Feedforward dimension: Hidden size in FFN after attention
        # Typically 2-4x d_model (Transformer paper uses 4x)
        'dim_feedforward': {'values': [128, 256, 512, 1024]},  # Removed extremes
        
        # Activation functions: For feedforward networks
        'activation': {
            'values': [
                'ReLU',   # Original Transformer paper default
                'GELU',   # Modern standard (BERT, GPT)
                'GLU',    # Gated Linear Unit
                'GEGLU',  # GELU-Gated (best of both worlds)
            ]
        },  # Removed Bilinear, ReGLU, SwiGLU (less tested in Darts)
        
        # Normalization type: Pre-norm vs post-norm architecture
        'norm_type': {
            'values': [
                'LayerNorm',        # Standard (post-norm)
                'RMSNorm',          # More stable, faster
                'LayerNormNoBias',  # Simpler variant
            ]
        },
        
        # Dropout: Critical for transformer regularization
        'dropout': {'values': [0.1, 0.2, 0.3, 0.4]},  # Removed 0.0, 0.5
        
        # Gradient clipping: Essential for attention stability
        'gradient_clip_val': {
            'distribution': 'uniform',
            'min': 0.5,
            'max': 1.5
        },
        
        # Loss Function Configuration
        'loss_function': {'values': ['WeightedPenaltyHuberLoss']},  # Best for fatalities
        
        # Zero threshold: What counts as "zero" in fatality data
        'zero_threshold': {
            'distribution': 'uniform',
            'min': 0.001,
            'max': 1.0
        },
        
        # False positives: Predicting conflict when there is none
        'false_positive_weight': {
            'distribution': 'uniform',
            'min': 2.0,
            'max': 20.0,
        },
        
        # False negatives: Missing actual conflicts (CRITICAL)
        'false_negative_weight': {
            'distribution': 'uniform',
            'min': 5.0,
            'max': 30.0,
        },
        
        # Non-zero weight: General importance of conflict events
        'non_zero_weight': {
            'distribution': 'uniform',
            'min': 3.0,
            'max': 20.0,
        },
        
        # Huber delta: Transition point between L2 and L1 loss
        'delta': {
            'distribution': 'uniform',
            'min': 0.01,
            'max': 5.0,
        },
    }
    sweep_config['parameters'] = parameters_dict

    return sweep_config