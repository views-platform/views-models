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

from tests.conftest import load_config_module

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
        hp = _load_hp(pf_model)
        n = hp.get("n_posterior_samples")
        assert isinstance(n, int) and n > 0, (
            f"{pf_model}: n_posterior_samples must be a positive int, got {n}"
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


class TestPFEEnsembleConfigReadiness:
    """Every PFE ensemble must reference valid PF constituent models."""

    pytestmark = [pytest.mark.green]

    @pytest.fixture(params=PFE_ENSEMBLES)
    def pfe_ensemble(self, request):
        return request.param

    def test_models_list_non_empty(self, pfe_ensemble):
        meta = _load_meta(pfe_ensemble, ENSEMBLES_DIR)
        models = meta.get("models")
        assert isinstance(models, list) and len(models) > 0

    def test_all_constituents_exist(self, pfe_ensemble):
        meta = _load_meta(pfe_ensemble, ENSEMBLES_DIR)
        for model_name in meta["models"]:
            assert (MODELS_DIR / model_name).is_dir(), (
                f"{pfe_ensemble} references {model_name} but it doesn't exist"
            )

    def test_all_constituents_are_pf_models(self, pfe_ensemble):
        meta = _load_meta(pfe_ensemble, ENSEMBLES_DIR)
        for model_name in meta["models"]:
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
        samples = []
        for model_name in meta["models"]:
            hp = _load_hp(model_name)
            samples.append(_require_n_posterior_samples(hp, model_name))
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
        expected_samples = _require_n_posterior_samples(hp, name)
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
        y = np.load(origin / target / "y_pred.npy", mmap_mode="r")
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
    samples = []
    for model_name in meta["models"]:
        hp = _load_hp(model_name)
        n = hp.get("n_posterior_samples")
        if n is None:
            pytest.skip(
                f"{ensemble_name}: constituent {model_name} missing "
                f"n_posterior_samples (green test catches this)"
            )
        samples.append(n)
    if aggregation == "concat":
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
