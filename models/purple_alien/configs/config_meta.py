def get_meta_config():
    """
    Contains the meta data for the model (model architecture, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    meta_config = {
        "name": "purple_alien",
        "algorithm": "HydraNet", 
        "targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"], #, "ln_sb_best_binarized", "ln_ns_best_binarized", "ln_os_best_binarized"], 
        # "queryset": "escwa001_cflong",
        "identity_cols" : ["priogrid_gid", "col", "row", "month_id", "c_id"],
        "index_names": ['month_id', 'priogrid_gid'],
        "features" : ["lr_sb_best", "lr_ns_best", "lr_os_best"],
        "level": "pgm",
        "creator": "Simon",
        "metrics": ["RMSLE", "CRPS", "MSE", "MSLE", "y_hat_bar", "AP"],
    }
    return meta_config 
