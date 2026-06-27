"""PredictionFrame ensemble production readiness tests.

TDD tests for the PFE production roadmap (views-pipeline-core
2026-06-01_pfe_production_roadmap.md). All expectations are derived
from model and ensemble configs — nothing is hardcoded.

Green tests (config-level) always run.
Red tests (output-level) skip when prediction outputs don't exist.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import load_config_module, regression_targets_by_location

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
ENSEMBLES_DIR = REPO_ROOT / "ensembles"


# ── Helpers ──────────────────────────────────────────────────────────

_GETTER_ALIASES = {"config_hyperparameters": "get_hp_config"}


def _load_config(base_dir, name, config_name):
    path = base_dir / name / "configs" / f"{config_name}.py"
    mod = load_config_module(path)
    getter = _GETTER_ALIASES.get(
        config_name, f"get_{config_name.removeprefix('config_')}_config"
    )
    return getattr(mod, getter)()


def _load_meta(name, base_dir=MODELS_DIR):
    return _load_config(base_dir, name, "config_meta")


def _load_hp(name, base_dir=MODELS_DIR):
    return _load_config(base_dir, name, "config_hyperparameters")


def _require_regression_targets(hp, name):
    targets = hp.get("regression_targets")
    if not targets:
        pytest.skip(f"{name}: no regression_targets in config (green test catches this)")
    return targets


def _require_n_posterior_samples(hp, name):
    n = hp.get("n_posterior_samples")
    if n is None:
        pytest.skip(f"{name}: no n_posterior_samples in config (green test catches this)")
    return n


def _model_eval_mode(name, base_dir=MODELS_DIR):
    """Resolve a PF model's content mode (``point``|``stochastic``) from config.

    The discriminator is ``config_meta.evaluation_mode`` (epic #216, decided in
    #217 — mirrors HydraNet's ``evaluation_mode``). A missing value defaults to
    ``stochastic`` so existing stochastic models are unaffected and only point
    models must opt in.
    """
    return _load_meta(name, base_dir).get("evaluation_mode", "stochastic")


def _n_posterior_samples_ok(mode, n):
    """Pure sampling-count predicate, branched on ``evaluation_mode`` (#216/#217).

    - ``point``  → the model does no sampling, so it must **omit**
      ``n_posterior_samples`` (``n is None``); declaring any count (even 1) is
      incoherent — that was the dishonest workaround this epic removes.
    - otherwise (``stochastic``) → ``n_posterior_samples`` must be a positive int.
    """
    if mode == "point":
        return n is None
    return isinstance(n, int) and n > 0


def _expected_output_width(name, base_dir=MODELS_DIR):
    """Expected ``y_pred`` sample-axis width for a single PF model (#216/#219).

    point ⇒ 1 (collapsed to a scalar per cell, à la HydraNet's
    ``collapse_to_point``); stochastic ⇒ ``n_posterior_samples`` (skip if absent
    — the green config test owns that failure).
    """
    if _model_eval_mode(name, base_dir) == "point":
        return 1
    hp = _load_hp(name, base_dir)
    return _require_n_posterior_samples(hp, name)


def _constituent_sample_count(model_name, ensemble_name=None):
    """Sample-axis columns contributed by one PF constituent (#216/#219).

    A point constituent contributes a single column; a stochastic constituent
    contributes its ``n_posterior_samples`` (skip if missing — the green config
    test owns that failure). Keeps ``concat`` (sum) and ``arithmetic_mean``
    expectations correct for ensembles mixing point + stochastic constituents.
    """
    if _model_eval_mode(model_name) == "point":
        return 1
    hp = _load_hp(model_name)
    n = hp.get("n_posterior_samples")
    if n is None:
        pytest.skip(
            f"{ensemble_name or model_name}: constituent {model_name} missing "
            f"n_posterior_samples (green test catches this)"
        )
    return n


def _discover_pf_models():
    """Return names of all models with prediction_format == 'prediction_frame'."""
    pf_models = []
    for d in sorted(MODELS_DIR.iterdir()):
        if not d.is_dir() or not (d / "configs" / "config_meta.py").exists():
            continue
        if d.name == "fake_model":
            continue
        try:
            meta = _load_meta(d.name)
            if meta.get("prediction_format") == "prediction_frame":
                pf_models.append(d.name)
        except (FileNotFoundError, AttributeError):
            continue
    return pf_models


def _discover_pfe_ensembles():
    """Return names of ensembles that use PredictionFrameEnsembleManager."""
    pfe = []
    for d in sorted(ENSEMBLES_DIR.iterdir()):
        main_py = d / "main.py"
        if not main_py.exists():
            continue
        text = main_py.read_text()
        if "PredictionFrameEnsembleManager" in text:
            pfe.append(d.name)
    return pfe


def _find_pf_prediction_runs(base_dir, name):
    """Find (run_type, timestamp) pairs with PF directory structure."""
    gen = base_dir / name / "data" / "generated"
    if not gen.exists():
        return []
    runs = []
    for d in sorted(gen.iterdir()):
        match = re.match(r"predictions_(\w+?)_(\d{8}_\d{6})$", d.name)
        if match and (d / "origin_0").is_dir():
            runs.append(match.groups())
    return runs


def _latest_pf_run(base_dir, name):
    """Return (run_type, timestamp) for the most recent PF run, or None."""
    runs = _find_pf_prediction_runs(base_dir, name)
    return runs[-1] if runs else None


PF_MODELS = _discover_pf_models()
PFE_ENSEMBLES = _discover_pfe_ensembles()


# ══════════════════════════════════════════════════════════════════════
# Issue #64 — Config-level readiness (green, always-run)
# ══════════════════════════════════════════════════════════════════════

class TestPFModelConfigReadiness:
    """Every PF model must have the config keys that PFE depends on."""

    pytestmark = [pytest.mark.green]

    @pytest.fixture(params=PF_MODELS)
    def pf_model(self, request):
        return request.param

    def test_has_n_posterior_samples(self, pf_model):
        """Sampling-count contract, branched on ``evaluation_mode`` (#216/#217).

        Point models do no sampling and must omit ``n_posterior_samples``;
        stochastic models must declare a positive int. Missing
        ``evaluation_mode`` defaults to stochastic.
        """
        hp = _load_hp(pf_model)
        n = hp.get("n_posterior_samples")
        mode = _model_eval_mode(pf_model)
        assert _n_posterior_samples_ok(mode, n), (
            f"{pf_model} (evaluation_mode={mode}): "
            + (
                f"point model must omit n_posterior_samples (does no sampling), got {n}"
                if mode == "point"
                else f"n_posterior_samples must be a positive int, got {n}"
            )
        )

    def test_has_prediction_format(self, pf_model):
        meta = _load_meta(pf_model)
        assert meta.get("prediction_format") == "prediction_frame"

    def test_has_regression_targets(self, pf_model):
        hp = _load_hp(pf_model)
        targets = hp.get("regression_targets")
        assert isinstance(targets, list) and len(targets) > 0, (
            f"{pf_model}: regression_targets must be a non-empty list"
        )

    def test_has_steps(self, pf_model):
        hp = _load_hp(pf_model)
        steps = hp.get("steps")
        assert isinstance(steps, list) and len(steps) > 0, (
            f"{pf_model}: steps must be a non-empty list"
        )

    def test_regression_targets_consistent_meta_hp(self, pf_model):
        # Delegates to the single source of truth (conftest). The repo-wide
        # agreement guard lives in tests/test_regression_targets.py (all models).
        located = regression_targets_by_location(MODELS_DIR / pf_model)
        if "meta" in located and "hp" in located:
            assert sorted(located["meta"]) == sorted(located["hp"]), (
                f"{pf_model}: config_meta regression_targets {located['meta']} != "
                f"config_hyperparameters regression_targets {located['hp']}"
            )
        else:
            pytest.skip(f"{pf_model}: regression_targets not declared in both configs")


class TestPFEEnsembleConfigReadiness:
    """Every PFE ensemble must reference valid PF constituent models."""

    pytestmark = [pytest.mark.green]

    @pytest.fixture(params=PFE_ENSEMBLES)
    def pfe_ensemble(self, request):
        return request.param

    def test_models_list_non_empty(self, pfe_ensemble):
        modelset = _load_config(ENSEMBLES_DIR, pfe_ensemble, "config_modelset")
        models = modelset.get("models")
        assert isinstance(models, list) and len(models) > 0

    def test_all_constituents_exist(self, pfe_ensemble):
        modelset = _load_config(ENSEMBLES_DIR, pfe_ensemble, "config_modelset")
        for model_name in modelset["models"]:
            assert (MODELS_DIR / model_name).is_dir(), (
                f"{pfe_ensemble} references {model_name} but it doesn't exist"
            )

    def test_all_constituents_are_pf_models(self, pfe_ensemble):
        modelset = _load_config(ENSEMBLES_DIR, pfe_ensemble, "config_modelset")
        for model_name in modelset["models"]:
            model_meta = _load_meta(model_name)
            assert model_meta.get("prediction_format") == "prediction_frame", (
                f"{pfe_ensemble} constituent {model_name} is not a PF model"
            )

    def test_aggregation_valid_for_pfe(self, pfe_ensemble):
        meta = _load_meta(pfe_ensemble, ENSEMBLES_DIR)
        assert meta.get("aggregation") in ("concat", "arithmetic_mean"), (
            f"{pfe_ensemble}: aggregation must be 'concat' or 'arithmetic_mean', "
            f"got '{meta.get('aggregation')}'"
        )

    def test_has_regression_targets(self, pfe_ensemble):
        meta = _load_meta(pfe_ensemble, ENSEMBLES_DIR)
        targets = meta.get("regression_targets")
        assert isinstance(targets, list) and len(targets) > 0

    def test_constituent_samples_consistent_for_mean(self, pfe_ensemble):
        meta = _load_meta(pfe_ensemble, ENSEMBLES_DIR)
        if meta.get("aggregation") != "arithmetic_mean":
            pytest.skip("only applies to arithmetic_mean aggregation")
        modelset = _load_config(ENSEMBLES_DIR, pfe_ensemble, "config_modelset")
        samples = [
            _constituent_sample_count(model_name, pfe_ensemble)
            for model_name in modelset["models"]
        ]
        assert len(set(samples)) == 1, (
            f"{pfe_ensemble} uses arithmetic_mean but constituents have "
            f"different n_posterior_samples: {samples}"
        )


# ══════════════════════════════════════════════════════════════════════
# Issue #65 — PredictionFrame output validation (red, skip-when-absent)
# ══════════════════════════════════════════════════════════════════════

def _pf_output_cases():
    """Yield (model_name, run_type, timestamp) for PF models with outputs."""
    cases = []
    for name in PF_MODELS:
        run = _latest_pf_run(MODELS_DIR, name)
        if run:
            cases.append((name, *run))
    return cases


PF_OUTPUT_CASES = _pf_output_cases()
PF_OUTPUT_IDS = [f"{c[0]}_{c[1]}" for c in PF_OUTPUT_CASES]


class TestPredictionFrameOutput:
    """Validate PF output structure against config-derived expectations."""

    pytestmark = [pytest.mark.red]

    @pytest.fixture(params=PF_OUTPUT_CASES, ids=PF_OUTPUT_IDS)
    def pf_case(self, request):
        return request.param

    def _pred_dir(self, pf_case):
        name, run_type, timestamp = pf_case
        return MODELS_DIR / name / "data" / "generated" / f"predictions_{run_type}_{timestamp}"

    def _origins(self, pf_case):
        pred_dir = self._pred_dir(pf_case)
        return sorted(
            d for d in pred_dir.iterdir()
            if d.is_dir() and d.name.startswith("origin_")
        )

    def test_all_regression_targets_have_directories(self, pf_case):
        name = pf_case[0]
        hp = _load_hp(name)
        targets = _require_regression_targets(hp, name)
        expected_targets = set(targets)
        origins = self._origins(pf_case)
        assert len(origins) > 0, f"{name}: no origin directories found"
        first_origin = origins[0]
        actual_targets = {d.name for d in first_origin.iterdir() if d.is_dir()}
        missing = expected_targets - actual_targets
        assert not missing, (
            f"{name}: origin_0 missing regression target dirs: {missing}"
        )

    def test_y_pred_shape(self, pf_case):
        name = pf_case[0]
        hp = _load_hp(name)
        targets = _require_regression_targets(hp, name)
        expected_samples = _expected_output_width(name)
        first_origin = self._origins(pf_case)[0]
        y = np.load(first_origin / targets[0] / "y_pred.npy", mmap_mode="r")
        assert y.ndim == 2, f"{name}: y_pred must be 2D, got {y.ndim}D"
        assert y.shape[0] > 0, f"{name}: y_pred has 0 rows"
        assert y.shape[1] == expected_samples, (
            f"{name}: y_pred.shape[1]={y.shape[1]} but "
            f"n_posterior_samples={expected_samples}"
        )

    def test_y_pred_dtype(self, pf_case):
        name = pf_case[0]
        hp = _load_hp(name)
        targets = _require_regression_targets(hp, name)
        first_origin = self._origins(pf_case)[0]
        y = np.load(first_origin / targets[0] / "y_pred.npy", mmap_mode="r")
        assert y.dtype in (np.float32, np.float64), (
            f"{name}: y_pred dtype must be float32/64, got {y.dtype}"
        )

    def test_identifiers_keys(self, pf_case):
        name = pf_case[0]
        hp = _load_hp(name)
        targets = _require_regression_targets(hp, name)
        first_origin = self._origins(pf_case)[0]
        ids = np.load(first_origin / targets[0] / "identifiers.npz")
        assert "time" in ids and "unit" in ids, (
            f"{name}: identifiers.npz must have 'time' and 'unit' keys, "
            f"got {list(ids.keys())}"
        )

    def test_identifiers_length_matches_predictions(self, pf_case):
        name = pf_case[0]
        hp = _load_hp(name)
        targets = _require_regression_targets(hp, name)
        first_origin = self._origins(pf_case)[0]
        y = np.load(first_origin / targets[0] / "y_pred.npy", mmap_mode="r")
        ids = np.load(first_origin / targets[0] / "identifiers.npz")
        assert len(ids["time"]) == y.shape[0], (
            f"{name}: time ids length {len(ids['time'])} != y_pred rows {y.shape[0]}"
        )
        assert len(ids["unit"]) == y.shape[0], (
            f"{name}: unit ids length {len(ids['unit'])} != y_pred rows {y.shape[0]}"
        )

    def test_origins_consistent_across_targets(self, pf_case):
        name = pf_case[0]
        hp = _load_hp(name)
        targets = _require_regression_targets(hp, name)
        pred_dir = self._pred_dir(pf_case)
        origin_sets = {}
        for target in targets:
            origins = {
                d.name for d in pred_dir.iterdir()
                if d.is_dir() and d.name.startswith("origin_")
                and (d / target).is_dir()
            }
            origin_sets[target] = origins
        target_names = list(origin_sets.keys())
        for t in target_names[1:]:
            assert origin_sets[target_names[0]] == origin_sets[t], (
                f"{name}: origin sets differ between {target_names[0]} and {t}"
            )


class TestTransformUndoScale:
    """Verify predictions are on measurement scale, not log-compressed."""

    pytestmark = [pytest.mark.red]

    @pytest.fixture(params=PF_OUTPUT_CASES, ids=PF_OUTPUT_IDS)
    def pf_case(self, request):
        return request.param

    def _first_target_origin(self, pf_case):
        name = pf_case[0]
        hp = _load_hp(name)
        targets = _require_regression_targets(hp, name)
        pred_dir = (
            MODELS_DIR / name / "data" / "generated"
            / f"predictions_{pf_case[1]}_{pf_case[2]}"
        )
        first_origin = sorted(
            d for d in pred_dir.iterdir()
            if d.is_dir() and d.name.startswith("origin_")
        )[0]
        return targets[0], first_origin

    def test_values_non_negative(self, pf_case):
        name = pf_case[0]
        target, origin = self._first_target_origin(pf_case)
        y = np.load(origin / target / "y_pred.npy", mmap_mode="r")
        assert y.min() >= 0, (
            f"{name}/{target}: min value {y.min():.4f} is negative — "
            f"fatality counts must be non-negative"
        )

    def test_values_not_log_compressed(self, pf_case):
        name = pf_case[0]
        target, origin = self._first_target_origin(pf_case)
        if target.startswith("synth_"):
            pytest.skip(f"{name}/{target}: synthetic target, log-compression check N/A")
        y = np.load(origin / target / "y_pred.npy", mmap_mode="r")
        if _load_meta(name)["algorithm"] == "ZeroModel":
            # C-76: a zero baseline correctly emits all-zeros — the max>10
            # heuristic is a false invariant for it. Assert the inverse:
            # a ZeroModel emitting nonzero is itself a bug.
            assert y.max() == 0 and y.min() == 0, (
                f"{name}/{target}: ZeroModel must predict all zeros, got "
                f"[{y.min():.4f}, {y.max():.4f}]"
            )
            return
        assert y.max() > 10, (
            f"{name}/{target}: max value {y.max():.4f} suggests log-scale — "
            f"measurement-scale fatality counts should exceed 10 in high-conflict cells"
        )

    def test_no_nan(self, pf_case):
        name = pf_case[0]
        target, origin = self._first_target_origin(pf_case)
        y = np.load(origin / target / "y_pred.npy", mmap_mode="r")
        nan_count = np.isnan(y).sum()
        assert nan_count == 0, f"{name}/{target}: {nan_count} NaN values"

    def test_no_inf(self, pf_case):
        name = pf_case[0]
        target, origin = self._first_target_origin(pf_case)
        y = np.load(origin / target / "y_pred.npy", mmap_mode="r")
        inf_count = np.isinf(y).sum()
        assert inf_count == 0, f"{name}/{target}: {inf_count} Inf values"


# ══════════════════════════════════════════════════════════════════════
# Issue #66 — PFE ensemble aggregation validation (red, skip-when-absent)
# ══════════════════════════════════════════════════════════════════════

def _pfe_output_cases():
    """Yield (ensemble_name, run_type, timestamp) for PFE ensembles with outputs."""
    cases = []
    for name in PFE_ENSEMBLES:
        run = _latest_pf_run(ENSEMBLES_DIR, name)
        if run:
            cases.append((name, *run))
    return cases


PFE_OUTPUT_CASES = _pfe_output_cases()
PFE_OUTPUT_IDS = [f"{c[0]}_{c[1]}" for c in PFE_OUTPUT_CASES]


def _expected_ensemble_samples(ensemble_name):
    meta = _load_meta(ensemble_name, ENSEMBLES_DIR)
    aggregation = meta["aggregation"]
    modelset = _load_config(ENSEMBLES_DIR, ensemble_name, "config_modelset")
    samples = [
        _constituent_sample_count(model_name, ensemble_name)
        for model_name in modelset["models"]
    ]
    if aggregation == "concat":
        # PFE concat = np.concatenate on the sample axis
        # (pipeline-core managers/ensemble/prediction_frame_ensemble.py:99), so the
        # pooled draw count is the SUM of constituent counts (empirically: synthetic_chant
        # 3×64 → 192; rusty_bucket 8×128 → 1024). golden_hour's 12-vs-40 is a separate
        # stale-artifact anomaly tracked as #131 / C-74 — not a contract-encoding error.
        return sum(samples)
    elif aggregation == "arithmetic_mean":
        return samples[0]
    return None


class TestPFEEnsembleAggregation:
    """Validate ensemble PF output: sample count, scale, integrity."""

    pytestmark = [pytest.mark.red]

    @pytest.fixture(params=PFE_OUTPUT_CASES if PFE_OUTPUT_CASES else [None],
                    ids=PFE_OUTPUT_IDS if PFE_OUTPUT_IDS else ["no_ensemble_output"])
    def pfe_case(self, request):
        if request.param is None:
            pytest.skip("no PFE ensemble predictions found")
        return request.param

    def _pred_dir(self, pfe_case):
        name, run_type, timestamp = pfe_case
        return ENSEMBLES_DIR / name / "data" / "generated" / f"predictions_{run_type}_{timestamp}"

    def _first_origin(self, pfe_case):
        pred_dir = self._pred_dir(pfe_case)
        return sorted(
            d for d in pred_dir.iterdir()
            if d.is_dir() and d.name.startswith("origin_")
        )[0]

    def test_all_targets_have_directories(self, pfe_case):
        name = pfe_case[0]
        meta = _load_meta(name, ENSEMBLES_DIR)
        expected_targets = set(meta["regression_targets"])
        first_origin = self._first_origin(pfe_case)
        actual_targets = {d.name for d in first_origin.iterdir() if d.is_dir()}
        missing = expected_targets - actual_targets
        assert not missing, (
            f"{name}: origin_0 missing regression target dirs: {missing}"
        )

    def test_aggregated_sample_count(self, pfe_case):
        name = pfe_case[0]
        meta = _load_meta(name, ENSEMBLES_DIR)
        expected = _expected_ensemble_samples(name)
        first_target = meta["regression_targets"][0]
        y = np.load(self._first_origin(pfe_case) / first_target / "y_pred.npy", mmap_mode="r")
        assert y.shape[1] == expected, (
            f"{name}: aggregated samples={y.shape[1]} but expected "
            f"{expected} (from constituent configs, aggregation='{meta['aggregation']}')"
        )

    def test_aggregated_values_non_negative(self, pfe_case):
        name = pfe_case[0]
        meta = _load_meta(name, ENSEMBLES_DIR)
        first_target = meta["regression_targets"][0]
        y = np.load(self._first_origin(pfe_case) / first_target / "y_pred.npy", mmap_mode="r")
        assert y.min() >= 0, (
            f"{name}/{first_target}: min={y.min():.4f} is negative"
        )

    def test_aggregated_values_not_log_compressed(self, pfe_case):
        name = pfe_case[0]
        meta = _load_meta(name, ENSEMBLES_DIR)
        first_target = meta["regression_targets"][0]
        if first_target.startswith("synth_"):
            pytest.skip(f"{name}/{first_target}: synthetic target, log-compression check N/A")
        y = np.load(self._first_origin(pfe_case) / first_target / "y_pred.npy", mmap_mode="r")
        assert y.max() > 10, (
            f"{name}/{first_target}: max={y.max():.4f} suggests log-scale"
        )

    def test_aggregated_no_nan_inf(self, pfe_case):
        name = pfe_case[0]
        meta = _load_meta(name, ENSEMBLES_DIR)
        first_target = meta["regression_targets"][0]
        y = np.load(self._first_origin(pfe_case) / first_target / "y_pred.npy", mmap_mode="r")
        assert np.isnan(y).sum() == 0, f"{name}: NaN in aggregated output"
        assert np.isinf(y).sum() == 0, f"{name}: Inf in aggregated output"

    def test_identifiers_present(self, pfe_case):
        name = pfe_case[0]
        meta = _load_meta(name, ENSEMBLES_DIR)
        first_target = meta["regression_targets"][0]
        ids = np.load(self._first_origin(pfe_case) / first_target / "identifiers.npz")
        assert "time" in ids and "unit" in ids, (
            f"{name}: identifiers.npz missing keys, got {list(ids.keys())}"
        )


# ══════════════════════════════════════════════════════════════════════
# Epic #216 — point/stochastic contract lockdown (positive + negative)
# ══════════════════════════════════════════════════════════════════════

class TestPointStochasticReadinessContract:
    """Pin the point/stochastic readiness contract so it cannot silently
    regress (#216/#221).

    A point model must OMIT ``n_posterior_samples``; a stochastic model must
    declare a positive int; a missing ``evaluation_mode`` defaults to
    stochastic. These tests are config/logic-level (no prediction artifacts).
    """

    pytestmark = [pytest.mark.green]

    # ── negative: point honesty — declaring any count (the old fake-1) is rejected
    @pytest.mark.parametrize("n,ok", [(None, True), (1, False), (5, False)])
    def test_point_must_omit_samples(self, n, ok):
        assert _n_posterior_samples_ok("point", n) is ok

    # ── negative: stochastic honesty — must declare a positive int
    @pytest.mark.parametrize("n,ok", [(None, False), (0, False), (1, True), (128, True)])
    def test_stochastic_requires_positive_int(self, n, ok):
        assert _n_posterior_samples_ok("stochastic", n) is ok

    # ── back-compat: a model with no evaluation_mode resolves to stochastic
    def test_missing_evaluation_mode_defaults_stochastic(self, monkeypatch):
        monkeypatch.setattr(
            "tests.test_pfe_production_readiness._load_meta",
            lambda name, base_dir=MODELS_DIR: {"prediction_format": "prediction_frame"},
        )
        assert _model_eval_mode("anything") == "stochastic"

    # ── output/aggregation: a point model contributes a single column
    def test_point_output_width_and_constituent_count_are_one(self, monkeypatch):
        monkeypatch.setattr(
            "tests.test_pfe_production_readiness._model_eval_mode",
            lambda *a, **k: "point",
        )
        assert _expected_output_width("any") == 1
        assert _constituent_sample_count("any", "ens") == 1

    # ── positive: real shipped point models pass honestly (re-faking is caught here)
    @pytest.mark.parametrize(
        "name", ["zero_cmbaseline", "locf_pgmbaseline", "diagonal_dream"]
    )
    def test_real_point_models_pass_without_samples(self, name):
        assert _model_eval_mode(name) == "point", (
            f"{name}: expected evaluation_mode=point in config_meta"
        )
        n = _load_hp(name).get("n_posterior_samples")
        assert n is None, (
            f"{name}: point model must not declare n_posterior_samples (got {n})"
        )
        assert _n_posterior_samples_ok("point", n) is True
