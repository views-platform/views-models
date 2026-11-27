def get_meta_config():
    """Meta for chrome_box — the pgm ensemble of epic #595, reconciled to its cm twin.

    ⚠️ `config_meta` is NOT documentation-only, whatever the scaffold template says. It is
    fourth in pipeline-core's five-source merge order (`configuration.py:460-492`) and
    silently overwrites `config_hyperparameters`; only `level`, `run_type` and
    `prediction_format` warn when it does (register C-68, C-175).

    ⛔ `reconciliation: "pgm_cm"`, not `"pgm_cm_point"`. The two are not interchangeable:
    `pgm_cm_point` is DEPRECATED (`DEPRECATED_RECONCILIATION_TYPES` in pipeline-core's
    sniffer) and does point reconciliation, which would collapse this ensemble's 1,024
    draws to a single number before reconciling. `pgm_cm` is the frames-native, draw-aware
    path, and it is the reason main.py uses PredictionFrameEnsembleManager — the DataFrame
    path REFUSES `pgm_cm` rather than implement it, and pipeline-core's own comment says
    the refusal exists to "protect the first one that does". This is that first one.

    ⚠️ No `evaluation_profile` key, deliberately. The sixteen members declare
    `hydranet_ucdp`, but `views_evaluation.profiles.PROFILES["hydranet_ucdp"]` is
    byte-identical to `["base"]`, the default when the key is absent — measured
    2026-10-27, every metric's hyperparameters equal. So declaring it changes no number,
    and it would make this ensemble subject to the C-132 occurrence-gate guard
    (`tests/test_roster_conformance.py`), which requires a 1:1 `by_*` channel. These
    members declare no classification targets at all — they are pure distributional
    regression baselines — so there is no occurrence channel to pool and declaring one
    would name a channel no member produces. Omitting the key is the honest form.

    Returns:
    - meta_config (dict): A dictionary containing ensemble meta configuration.
    """
    meta_config = {
        "name": "chrome_box",
        "regression_targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
        "level": "pgm",
        # Pools 8 members x 128 draws = 1,024 on the sample axis. "concat" is the
        # distributional aggregation; "mean"/"median" would discard the posterior that is
        # the whole point of epic #595.
        "aggregation": "concat",
        # What the sixteen members declare, so the pool is scored the way its parts are.
        "regression_sample_metrics": ["CRPS", "QS_sample", "MCR_sample", "Brier_rgs_sample"],
        "reconciliation": "pgm_cm",
        "reconcile_with": "chrome_crate",
        "creator": "Simon",
    }
    return meta_config
