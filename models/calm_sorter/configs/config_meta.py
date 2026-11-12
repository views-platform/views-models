def get_meta_config():
    """
    Contains the meta data for the model (model architecture, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    meta_config = {
        # ============================================================
        # General information
        # ============================================================
        "name": "calm_sorter",
        "algorithm": "HydraNet", 
        "creator": "Simon",
        "level": "pgm",
        
        # ============================================================
        # output format
        # ============================================================

        "prediction_format": "prediction_frame",
        # ============================================================
        # diagnostic settings
        # ============================================================
        "diagnostic_visualizations": False,  # was False

        # ============================================================
        # evaluation settings 
        # ============================================================
        # ⛔ THESE THREE ARE MAGNITUDE METRICS AND THEY ARE VOID ON THE `*_sorter` ARM —
        # not weak, void. Under the listwise objective the score is translation-invariant
        # (Bai2023 Prop. 1): it fits ORDER and leaves LEVEL unconstrained, and
        # `output_distribution: "standard"` then writes that arbitrarily-scaled score out as
        # if it were a fatality count. So CRPS / QS / MCR / coverage / calibration on a sorter
        # are confident numbers about nothing. Only RANK readouts are valid there — PAI@k,
        # NDCG@k, the AUC-over-cut-point curve, pairwise concordance (views-hydranet's
        # evaluation report battery). A delivery check that flags an implausibly large maximum
        # can also fire on a perfectly healthy sorter, because the scale is free to drift.
        # ⚠️ KEPT IDENTICAL IN BOTH ARMS DELIBERATELY. Stripping them from the sorter alone
        # would make the pair differ in something other than `loss_reg`, which is the one
        # invariant #729 rests on. The fix is to ignore them on the sorter, not to remove them.
        "regression_sample_metrics": ["CRPS", "QS_sample", "MCR_sample"],
        "classification_sample_metrics": ["Brier_cls_sample"],
        "evaluation_profile": "hydranet_ucdp",
        "rolling_origin_stride": 1,
    }
    return meta_config 
