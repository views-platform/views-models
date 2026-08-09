def get_meta_config():
    meta_config = {
        "name": "rusty_bucket",
        "regression_targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
        # Declare the occurrence/gate channel so the concat pool carries it (C-132):
        # `_build_context` derives its target list via `combined_targets`
        # (regression + classification), so the members' `by_*` gate PFs are pooled
        # alongside the `lr_*` magnitudes — restoring the ensemble's calibrated
        # occurrence/AP. (Supersedes the transient `targets` gate-hack, which the
        # #380 retirement now refuses.)
        "classification_targets": ["by_sb_best", "by_ns_best", "by_os_best"],
        "level": "pgm",
        "aggregation": "concat",
        "regression_sample_metrics": ["CRPS", "QS_sample", "MCR_sample"],
        "evaluation_profile": "hydranet_ucdp",
        "creator": "Simon",
        "reconciliation": None,
    }
    return meta_config
