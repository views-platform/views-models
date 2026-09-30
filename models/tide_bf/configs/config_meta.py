def get_meta_config():
    """Metadata for the base-feature TiDE GDP-per-capita disaggregation model."""
    return {
        "name": "tide_bf",
        "algorithm": "TiDEModel",
        "level": "cm",
        "regression_targets": ["lr_gdp_pcap"],
        "regression_point_metrics": ["MSE", "MSLE", "y_hat_bar", "MCR_point"],
        "creator": "Dylan",
        "time_steps": 36,
        "evaluation_sequencing": "horizon_chunks",
        "prediction_format": "prediction_frame",
        "rolling_origin_stride": 1,
        "skip_predictions_delivery": True,
    }
