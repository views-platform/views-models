def get_meta_config():
    """Metadata for the conflict-feature TiDE GDP-per-capita disaggregation model."""
    return {
        "name": "tide_bfc",
        "algorithm": "TiDEModel",
        "level": "cm",
        "regression_targets": ["lr_gdp_pcap"],
        "regression_point_metrics": ["MCR_point", "MSE", "MSLE", "y_hat_bar"],
        "creator": "Dylan",
        "time_steps": 36,
        "evaluation_sequencing": "horizon_chunks",
        "prediction_format": "prediction_frame",
        "rolling_origin_stride": 1,
        "skip_predictions_delivery": True,
    }
