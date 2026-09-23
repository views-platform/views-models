def get_meta_config():
    """
    Contains the meta data for the model (model algorithm, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    
    meta_config = {
        "name": "blue_ocean",
        "algorithm": "NBEATSModel",
        # Uncomment and modify the following lines as needed for additional metadata:
        "regression_targets": ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"],
        "level": "pgm",
        # The entity dimension these predictions are indexed by. views-r2darts2 defaults it to
        # "country_id" (darts_forecasting_model_manager.py:140) whatever the level, which is right
        # for the 31 cm models and wrong for these: pipeline-core's CorePredictionSniffer expects
        # {priogrid_id, month_id} at pgm and refuses {month_id, country_id}. The VALUES were always
        # priogrid cells (64,818 of them) — only the label was wrong. views-r2darts2#55, and
        # views-pipeline-core#529 is why the refusal was invisible.
        "entity_id": "priogrid_id",
        "creator": "Dylan",
        "regression_point_baselines": ["average_pgmbaseline", "zero_pgmbaseline", "locf_pgmbaseline"],
        "regression_point_metrics": ["MCR_point", "MSE", "MSLE", "y_hat_bar"],
        # "regression_sample_metrics": ["CRPS", "y_hat_bar", "twCRPS", "QIS", "MIS", "MCR_sample"],
        # "regression_sample_baselines": ["red_ranger"],
        "rolling_origin_stride": 1,
        "prediction_format": "dataframe",
        "skip_predictions_delivery": True,
    }
    return meta_config
