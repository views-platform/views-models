def get_meta_config():
    meta_config = {
        "name": "rusty_bucket",
        "regression_targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
        # The occurrence/gate channel, so the concat pool carries it (C-132).
        # views-pipeline-core#422 (in 3.0.1) derives the pooled target list via
        # `combined_targets` = regression + classification, so the members' `by_*` gate
        # PFs are pooled alongside the `lr_*` magnitudes. Without this declaration the
        # pool silently drops occurrence and the ensemble's AP is understated with no
        # error anywhere.
        #
        # All three lines below land together, and the split between them is not
        # cosmetic. Declaring `classification_targets` with NO classification metric key
        # is refused at load by `CoreConfigSniffer._check_targets_and_metrics` — the
        # defect PR #367 shipped. And `AP` belongs under **point**: views-models#372
        # originally advised the sample key, which passes the sniffer and then fails
        # `views_evaluation.NativeEvaluator._validate_config`, because METRIC_MEMBERSHIP
        # puts AP in ("classification", "point"). That would move the failure from config
        # load to evaluation time — later and quieter (their C-287).
        #
        # `Brier_cls_sample` is additionally what all eight constituents declare.
        "classification_targets": ["by_sb_best", "by_ns_best", "by_os_best"],
        "level": "pgm",
        "aggregation": "concat",
        "regression_sample_metrics": ["CRPS", "QS_sample", "MCR_sample"],
        "classification_point_metrics": ["AP"],
        "classification_sample_metrics": ["Brier_cls_sample"],
        "evaluation_profile": "hydranet_ucdp",
        "creator": "Simon",
        "reconciliation": None,
    }
    return meta_config
