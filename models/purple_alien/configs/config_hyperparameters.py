
def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.
    This configuration is "operational" so modifying these settings will impact the model's behavior during training.

    Returns:
    - hyperparameters (dict): A dictionary containing hyperparameters for training the model,
      which determine the model's behavior during the training phase.
    """

    hyperparameters = {

        # ============================================================
        # Ledger / Topology (ADR 007 Compliance)
        # ============================================================
        'time_col': 'month_id',
        'id_col': 'priogrid_gid',
        'spatial_cols': ['row', 'col'],
        'identity_cols': ['month_id', 'priogrid_gid', 'c_id', 'row_id', 'col_id'],
        'features': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
        'input_channels': 3, # Checksum: Must match len(features)
        'row_offset': 87,
        'col_offset': 310,
        'height': 180,
        'width': 180,

        # ============================================================
        # Model Architecture
        # ============================================================
        'model': 'HydraBNUNet06_LSTM4',
        'total_hidden_channels': 32,
        'dropout_rate': 0.125,
        'window_dim': 32,
        'output_channels': 1, # Depth per head
        'weight_init': 'xavier_norm',
        'freeze_h': "hl",
        'h_init': 'abs_rand_exp-100',

        # ============================================================
        # Optimization (ADR 014 Compliance)
        # ============================================================
        'windows_per_lesson': 3,     
        'learning_rate': 0.001,
        'weight_decay': 0.1,
        'scheduler': 'WarmupDecay',
        'warmup_steps': 100,
        'clip_grad_norm': True,
        'torch_seed': 4,
        'np_seed': 4,

        # ============================================================
        # Multi-Task Signals (ADR 020 Compliance)
        # ============================================================
        #'target_variable': 'lr_sb_best',
        'classification_targets': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'], # auto transform to by_ 
        'regression_targets': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
        'transform': {
            'log1p': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
            'asinh': [],
            'identity': []
        },
        'steps': list(range(1, 37)),
        'time_steps': 36, # Checksum: Must match len(steps)

        # ============================================================
        # Loss Functions
        # ============================================================
        'loss_reg': 'b',
        'loss_class': 'b',
        'loss_reg_a': 258,
        'loss_reg_c': 0.001,
        'loss_class_alpha': 0.75,
        'loss_class_gamma': 1.5,

        # ============================================================
        # Strategy (Curriculum ADR 011/012 Compliance)
        # ============================================================
        'total_lessons': 300,        
        'max_ratio': 0.95,           
        'min_ratio': 0.05,           
        'slope_ratio': 0.75,         
        'roof_ratio': 0.7,           
        'min_events': 5,             

        # ============================================================
        # Outbound / Evaluation
        # ============================================================
        # Note: Internal Naming (pred_, _raw, _prob) is handled by VolumeHandler
        'n_posterior_samples': 3,
        'evalution_mode': "point", #'stochastic',
        'aggregate_method': 'arithmetic_mean',
        'run_type': 'calibration',
    }

    return hyperparameters

