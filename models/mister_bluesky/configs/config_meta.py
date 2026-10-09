def get_meta_config():
    """
    Contains the meta data for the model (model algorithm, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    
    meta_config = {
        "name": "mister_bluesky", 
        "algorithm": "TSMixerModel",
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
        # #536: point metrics re-activated because they are now REQUIRED, not preferred.
        # views-evaluation picks the metric list from the DATA, not the config
        # (native_evaluator.py:258, `"sample" if n_samples > 1 else "point"`), so at
        # num_samples=1 it reads regression_point_metrics — and an empty list raises AFTER
        # the full training run, writing no predictions. The sample metrics below are kept
        # deliberately: they record what these two models are FOR, they are never read at one
        # sample, and keeping them makes the #492 revert a two-line change rather than six.
        "regression_point_baselines": ["average_pgmbaseline", "zero_pgmbaseline", "locf_pgmbaseline"],
        "regression_point_metrics": ["MCR_point", "MSE", "MSLE", "y_hat_bar"],
        "regression_sample_metrics": ["y_hat_bar", "twCRPS", "QIS", "MIS", "MCR_sample", "CRPS"],
        "regression_sample_baselines": ["black_ranger", "blue_ranger", "pink_ranger", "white_ranger"],
        "rolling_origin_stride": 1,
        "prediction_format": "dataframe",
        "skip_predictions_delivery": True,
    }
    return meta_config
