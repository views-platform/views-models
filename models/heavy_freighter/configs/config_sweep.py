def get_sweep_config():

    """
    Contains the configuration for hyperparameter sweeps using WandB.
    This configuration is "operational" so modifying it will change the search strategy, parameter ranges, and other settings for hyperparameter tuning aimed at optimizing model performance.

    Returns:
    - sweep_config (dict): A dictionary containing the configuration for hyperparameter sweeps, defining the methods and parameter ranges used to search for optimal hyperparameters.
    """

    sweep_config = {
    'name': 'heavy_freighter_sweep',
    'method': 'grid'
    }

    metric = {
        'name': '36month_mean_squared_error',
        'goal': 'minimize'
        }

    sweep_config['metric'] = metric

    parameters_dict = {
        'model' : {'value' :'HydraBNUNet06_LSTM4'},
        'weight_init' : {'value' : 'xavier_norm'}, # ['xavier_uni', 'xavier_norm', 'kaiming_uni', 'kaiming_normal']
        'clip_grad_norm' : {'value': True},
        'scheduler' : {'value': 'WarmupDecay'}, #CosineAnnealingLR004  'CosineAnnealingLR' 'OneCycleLR'
        'total_hidden_channels': {'value': 32}, # you like need 32, it seems from qualitative results
        'min_events': {'value': 5},
        'windows_per_lesson': {'value': 3},
        'total_lessons': {'value': 150},
        'batch_size': {'value':  3}, # just speed running here..
        "dropout_rate" : {'value' : 0.125},
        'learning_rate': {'value' :  0.001}, #0.001 default, but 0.005 might be better
        "weight_decay" : {'value' : 0.1},
        "slope_ratio" : {'value' : 0.75},
        "roof_ratio" : {'value' :  0.7},
        "max_ratio" : {'value' :  0.95},
        "min_ratio" : {'value' :  0.05},
        'input_channels' : {'value' : 3},
        'output_channels': {'value' : 1},
        'classification_targets': {'value': ['by_sb_best', 'by_ns_best', 'by_os_best']},
        'regression_targets': {'value': ['lr_sb_best', 'lr_ns_best', 'lr_os_best']},
        'loss_class' : { 'value' : 'b'}, # det nytter jo ikke noget at du køre over gamma og alpha for loss-class a...
        'loss_class_gamma' : {'value' : 1.5},
        'loss_class_alpha' : {'value' : 0.75}, # should be between 0.5 and 0.95...
        'loss_reg' : { 'value' :  'b'},
        'loss_reg_a' : { 'value' : 258},
        'loss_reg_c' : { 'value' : 0.001},
        'np_seed' : {'values' : [4, 8]},
        'torch_seed' : {'values' : [4, 8]},
        'window_dim' : {'value' : 32},
        'h_init' : {'value' : 'abs_rand_exp-100'},
        'warmup_steps' : {'value' : 100},
        'freeze_h' : {'value' : "hl"},
        'time_steps' : {'value' : 36}
        }

    sweep_config['parameters'] = parameters_dict

    return sweep_config
