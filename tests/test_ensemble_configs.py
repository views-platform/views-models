"""Tests for ensemble configuration completeness and dependency validation.

Ensembles have different required config keys than individual models:
- config_meta.py: name, regression_targets, level, aggregation
- config_modelset.py: models (list of constituent model names)
- config_deployment.py: deployment_status
- config_hyperparameters.py: steps
- config_partitions.py: generate() function

Ensembles do NOT have config_sweep.py or config_queryset.py.
"""
import pytest

from tests.conftest import (
    get_n_posterior_samples,
    get_regression_targets,
    load_config_module,
    MODELS_DIR,
    ENSEMBLES_DIR,
)


pytestmark = pytest.mark.beige

REQUIRED_ENSEMBLE_META_KEYS = {"name", "regression_targets", "level", "aggregation"}

REQUIRED_ENSEMBLE_CONFIG_FILES = [
    "config_meta.py",
    "config_modelset.py",
    "config_deployment.py",
    "config_hyperparameters.py",
    "config_partitions.py",
]

VALID_DEPLOYMENT_STATUSES = {"shadow", "deployed", "baseline", "deprecated"}


# ── Structure ──────────────────────────────────────────────────────────

class TestEnsembleStructure:
    def test_main_py_exists(self, ensemble_dir):
        assert (ensemble_dir / "main.py").exists()

    def test_run_sh_exists(self, ensemble_dir):
        assert (ensemble_dir / "run.sh").exists()

    def test_configs_directory_exists(self, ensemble_dir):
        assert (ensemble_dir / "configs").is_dir()

    @pytest.mark.parametrize("config_file", REQUIRED_ENSEMBLE_CONFIG_FILES)
    def test_required_config_file_exists(self, ensemble_dir, config_file):
        cfg_path = ensemble_dir / "configs" / config_file
        assert cfg_path.exists(), (
            f"{ensemble_dir.name} missing config file: {config_file}"
        )


# ── Config Completeness ───────────────────────────────────────────────

class TestEnsembleConfigMeta:
    def test_meta_has_required_keys(self, ensemble_dir):
        cfg_path = ensemble_dir / "configs" / "config_meta.py"
        module = load_config_module(cfg_path)
        meta = module.get_meta_config()
        missing = REQUIRED_ENSEMBLE_META_KEYS - set(meta.keys())
        assert not missing, (
            f"{ensemble_dir.name} config_meta missing keys: {missing}"
        )

    def test_meta_name_matches_directory(self, ensemble_dir):
        cfg_path = ensemble_dir / "configs" / "config_meta.py"
        module = load_config_module(cfg_path)
        meta = module.get_meta_config()
        assert meta["name"] == ensemble_dir.name

    def test_meta_level_is_valid(self, ensemble_dir):
        cfg_path = ensemble_dir / "configs" / "config_meta.py"
        module = load_config_module(cfg_path)
        meta = module.get_meta_config()
        assert meta["level"] in ("cm", "pgm")

    def test_no_old_targets_key(self, ensemble_dir):
        """Ensembles must use 'regression_targets', not the old 'targets' key."""
        cfg_path = ensemble_dir / "configs" / "config_meta.py"
        module = load_config_module(cfg_path)
        meta = module.get_meta_config()
        assert "targets" not in meta, (
            f"{ensemble_dir.name} still has old 'targets' key"
        )

    def test_no_old_metrics_key(self, ensemble_dir):
        """Ensembles must use 'regression_point_metrics', not old 'metrics' key."""
        cfg_path = ensemble_dir / "configs" / "config_meta.py"
        module = load_config_module(cfg_path)
        meta = module.get_meta_config()
        assert "metrics" not in meta, (
            f"{ensemble_dir.name} still has old 'metrics' key"
        )


class TestEnsembleConfigDeployment:
    def test_deployment_has_valid_status(self, ensemble_dir):
        cfg_path = ensemble_dir / "configs" / "config_deployment.py"
        module = load_config_module(cfg_path)
        dep = module.get_deployment_config()
        assert dep.get("deployment_status") in VALID_DEPLOYMENT_STATUSES, (
            f"{ensemble_dir.name} has invalid deployment_status"
        )


class TestEnsembleConfigHyperparameters:
    def test_hp_has_steps(self, ensemble_dir):
        cfg_path = ensemble_dir / "configs" / "config_hyperparameters.py"
        module = load_config_module(cfg_path)
        hp = module.get_hp_config()
        assert "steps" in hp, f"{ensemble_dir.name} config_hp missing 'steps'"


class TestEnsembleConfigModelset:
    def test_modelset_has_models_key(self, ensemble_dir):
        cfg_path = ensemble_dir / "configs" / "config_modelset.py"
        module = load_config_module(cfg_path)
        modelset = module.get_modelset_config()
        assert "models" in modelset, (
            f"{ensemble_dir.name} config_modelset missing 'models' key"
        )

    def test_modelset_models_is_nonempty_list(self, ensemble_dir):
        cfg_path = ensemble_dir / "configs" / "config_modelset.py"
        module = load_config_module(cfg_path)
        modelset = module.get_modelset_config()
        assert isinstance(modelset["models"], list) and len(modelset["models"]) > 0, (
            f"{ensemble_dir.name} config_modelset.models must be a non-empty list"
        )


# ── Dependency Validation ─────────────────────────────────────────────

class TestEnsembleDependencies:
    def test_all_constituent_models_exist(self, ensemble_dir):
        """Every model listed in config_modelset.models must exist as a model directory."""
        cfg_path = ensemble_dir / "configs" / "config_modelset.py"
        module = load_config_module(cfg_path)
        modelset = module.get_modelset_config()
        missing = [
            m for m in modelset["models"]
            if not (MODELS_DIR / m).is_dir()
        ]
        assert not missing, (
            f"{ensemble_dir.name} references non-existent models: {missing}"
        )

    def test_constituent_models_match_ensemble_level(self, ensemble_dir):
        """All models in an ensemble must have the same level (cm/pgm) as the ensemble."""
        meta_path = ensemble_dir / "configs" / "config_meta.py"
        meta_module = load_config_module(meta_path)
        meta = meta_module.get_meta_config()
        ensemble_level = meta["level"]

        modelset_path = ensemble_dir / "configs" / "config_modelset.py"
        modelset_module = load_config_module(modelset_path)
        modelset = modelset_module.get_modelset_config()

        mismatched = []
        for model_name in modelset["models"]:
            model_meta_path = MODELS_DIR / model_name / "configs" / "config_meta.py"
            if model_meta_path.exists():
                model_module = load_config_module(model_meta_path)
                model_level = model_module.get_meta_config().get("level")
                if model_level != ensemble_level:
                    mismatched.append(f"{model_name} (level={model_level})")
        assert not mismatched, (
            f"{ensemble_dir.name} is level='{ensemble_level}' but contains "
            f"models with different levels: {mismatched}"
        )

    def test_constituents_cover_ensemble_targets(self, ensemble_dir):
        """Every constituent must declare (and so produce) all of the ensemble's
        regression_targets.

        The ensemble pools constituent predictions BY target name — it looks for
        each declared target under each constituent's output and fails at run time
        if one is missing. This is the ONE place target-name agreement is
        structurally required (making it removable is pipeline-core
        views-pipeline-core#203). Config-derived via ``conftest.get_regression_targets``
        — no hardcoded target-name literal (EPIC #154 / S6).
        """
        meta_module = load_config_module(ensemble_dir / "configs" / "config_meta.py")
        ensemble_targets = set(meta_module.get_meta_config().get("regression_targets") or [])
        if not ensemble_targets:
            pytest.skip(f"{ensemble_dir.name} declares no regression_targets")

        modelset_module = load_config_module(ensemble_dir / "configs" / "config_modelset.py")
        constituents = modelset_module.get_modelset_config().get("models", [])

        gaps = {}
        for model_name in constituents:
            model_dir = MODELS_DIR / model_name
            if not model_dir.is_dir():
                continue  # existence is covered by test_all_constituent_models_exist
            missing = ensemble_targets - set(get_regression_targets(model_dir))
            if missing:
                gaps[model_name] = sorted(missing)
        assert not gaps, (
            f"{ensemble_dir.name} declares regression_targets {sorted(ensemble_targets)} but "
            f"these constituents do not produce all of them: {gaps} — the ensemble pools by "
            f"target name and would fail at run time"
        )

    def test_reconcile_with_target_exists(self, ensemble_dir):
        """If reconcile_with is declared, the target must exist as an ensemble."""
        cfg_path = ensemble_dir / "configs" / "config_meta.py"
        module = load_config_module(cfg_path)
        meta = module.get_meta_config()
        target = meta.get("reconcile_with")
        if target is not None:
            assert (ENSEMBLES_DIR / target).is_dir(), (
                f"{ensemble_dir.name} declares reconcile_with='{target}' "
                f"but no ensemble directory '{target}' exists"
            )

    def test_pgm_cm_point_ensemble_wires_a_reconciler(self, ensemble_dir):
        """An ensemble declaring reconciliation: 'pgm_cm_point' MUST inject a
        reconciler in its main.py, or the run fails loud at the seam
        (RECONCILER_NOT_INJECTED). This catches a new unwired reconciling ensemble
        at CI instead of at the monthly run (EPIC #172 / ADR-014) — the wired/
        unwired state is declared and visible, never accidental.
        """
        meta = load_config_module(
            ensemble_dir / "configs" / "config_meta.py"
        ).get_meta_config()
        if meta.get("reconciliation") != "pgm_cm_point":
            pytest.skip(f"{ensemble_dir.name} does not declare pgm_cm_point reconciliation")
        main_text = (ensemble_dir / "main.py").read_text()
        assert "build_reconciler" in main_text and "reconciler=" in main_text, (
            f"{ensemble_dir.name} declares reconciliation='pgm_cm_point' but its main.py "
            f"does not wire a reconciler (expected build_reconciler(...) + reconciler=) — "
            f"it will fail loud at runtime. See EPIC #172 / ADR-014."
        )

    def test_declared_modelset_and_sample_counts_match_reality(self, ensemble_dir):
        """Opt-in belt-and-suspenders contract (ADR-015): an ensemble that declares
        ``expected_models`` / ``expected_samples_per_model`` in config_hyperparameters
        must have those numbers match reality —

          (a) expected_models == len(config_modelset["models"]), and
          (b) every constituent declares n_posterior_samples == expected_samples_per_model.

        concat pools by resampling to a FIXED size (= each constituent's count) and
        pipeline-core's aggregator hard-requires equal counts (modules/aggregation/
        aggregator.py:627), so a single declared number is correct. This pulls that
        check forward to CI — a mismatch fails in seconds, not at minute-45 of a run.
        Ensembles that do not declare the fields are skipped (legacy-compatible).
        """
        hp = load_config_module(
            ensemble_dir / "configs" / "config_hyperparameters.py"
        ).get_hp_config()
        expected_models = hp.get("expected_models")
        expected_samples = hp.get("expected_samples_per_model")
        if expected_models is None and expected_samples is None:
            pytest.skip(f"{ensemble_dir.name} declares no expected_models/samples contract")

        models = load_config_module(
            ensemble_dir / "configs" / "config_modelset.py"
        ).get_modelset_config().get("models", [])

        if expected_models is not None:
            assert expected_models == len(models), (
                f"{ensemble_dir.name}: config declares expected_models={expected_models} "
                f"but config_modelset lists {len(models)} ({models}) — declare them "
                f"explicitly and keep them in sync (ADR-015)."
            )

        if expected_samples is not None:
            mismatches = {}
            for model_name in models:
                model_dir = MODELS_DIR / model_name
                if not model_dir.is_dir():
                    continue  # existence covered by test_all_constituent_models_exist
                n = get_n_posterior_samples(model_dir)
                if n != expected_samples:
                    mismatches[model_name] = n
            assert not mismatches, (
                f"{ensemble_dir.name}: declares expected_samples_per_model="
                f"{expected_samples} but these constituents declare a different "
                f"n_posterior_samples: {mismatches} — concat requires equal counts "
                f"(pipeline-core aggregator.py:627); normalize them (ADR-015)."
            )
