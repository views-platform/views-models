"""Tests for ensemble configuration completeness and dependency validation.

Ensembles have different required config keys than individual models:
- config_meta.py: name, regression_targets, level, aggregation
- config_modelset.py: models (list of constituent model names)
- config_maturity.py: maturity  (or the legacy config_deployment.py: deployment_status)
- config_hyperparameters.py: steps
- config_partitions.py: generate() function

Ensembles do NOT have config_sweep.py or config_queryset.py.
"""
import ast

import pytest

from tests.conftest import (
    get_produced_sample_count,
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
    "config_hyperparameters.py",
    "config_partitions.py",
]

VALID_MATURITIES = {"candidate", "graduate", "retired"}
VALID_DEPLOYMENT_STATUSES = {"shadow", "deployed", "baseline", "deprecated"}

#: Ensembles that declare a reconciliation mode pipeline-core has deprecated. ⛔ This set
#: MAY ONLY SHRINK. Both entries predate the deprecation and migrating them is
#: pipeline-core's own stated removal path, tracked separately — they are listed by name
#: so that the rule applies to everything else, rather than the rule being dropped to
#: accommodate them. Adding a name here is declaring new debt and needs the issue to go
#: with it.
RECONCILIATION_DEPRECATION_DEBT = frozenset({"white_mustang", "skinny_love"})


def _called_name(call):
    """The name being called in an `ast.Call`, or None for a shape we do not read."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


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


class TestEnsembleMaturityConfig:
    """Exactly one maturity file per ensemble, in one of the two vocabularies (ADR-017
    Phase 2; the #455 guard). Mirrors tests/test_config_completeness.py::TestMaturityConfig
    for models — a deliberate second copy, because the two fixtures differ."""

    def test_exactly_one_maturity_file(self, ensemble_dir):
        configs = ensemble_dir / "configs"
        new, legacy = configs / "config_maturity.py", configs / "config_deployment.py"
        assert not (new.exists() and legacy.exists()), (
            f"{ensemble_dir.name} carries BOTH config_maturity.py and config_deployment.py. "
            f"ADR-017 Phase 2 is a rename; delete the legacy file. (#455)"
        )
        assert new.exists() or legacy.exists(), (
            f"{ensemble_dir.name} has neither config_maturity.py nor config_deployment.py"
        )

    def test_maturity_value_is_valid(self, ensemble_dir):
        configs = ensemble_dir / "configs"
        if (configs / "config_maturity.py").exists():
            value = load_config_module(configs / "config_maturity.py").get_maturity_config().get("maturity")
            assert value in VALID_MATURITIES, f"{ensemble_dir.name} has invalid maturity: '{value}'"
        else:
            value = load_config_module(configs / "config_deployment.py").get_deployment_config().get("deployment_status")
            assert value in VALID_DEPLOYMENT_STATUSES, f"{ensemble_dir.name} has invalid deployment_status: '{value}'"


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

    def test_a_reconciling_ensemble_wires_a_reconciler(self, ensemble_dir):
        """An ensemble declaring ANY reconciliation MUST inject a reconciler in its
        main.py, or the run fails loud at the seam (RECONCILER_NOT_INJECTED). This
        catches a new unwired reconciling ensemble at CI instead of at the monthly run
        (EPIC #172 / ADR-014) — the wired/unwired state is declared and visible, never
        accidental.

        ⛔ This used to skip unless the value was the literal `"pgm_cm_point"`, which made
        it blind to the first ensemble declaring `"pgm_cm"` — epic #595's `chrome_box`.
        pipeline-core deprecated `pgm_cm_point` and recommends `pgm_cm`, so the guard was
        keyed on the value being retired and would have silently stopped applying exactly
        as the fleet moved. Every reconciliation needs a reconciler regardless of mode, so
        the condition is now "is one declared", which cannot go stale with the vocabulary.
        """
        meta = load_config_module(
            ensemble_dir / "configs" / "config_meta.py"
        ).get_meta_config()
        reconciliation = meta.get("reconciliation")
        if not reconciliation:
            pytest.skip(f"{ensemble_dir.name} declares no reconciliation")
        # ⛔ Read from the AST, not from the source text. Two text forms of this guard
        # have now been defeated, both in the same way — a pattern cannot tell whether
        # what it matched is code:
        #
        #   `"build_reconciler" in main_text`              — satisfied by the import line.
        #   `re.search(r"reconciler\s*=\s*build_...\(")`   — satisfied by a `#` comment,
        #                                                    with `reconciler = None` live.
        #
        # Widening a string pattern replaced one evasion with another in the same class,
        # which is the lesson: a tighter string is not a different kind of guard. A
        # comment has no AST node, so the first is gone, and the position check closes
        # the other half — moving the build BELOW the manager call is a
        # use-before-assignment at run time, and the text form could not see order.
        tree = ast.parse((ensemble_dir / "main.py").read_text(encoding="utf-8"))
        builds = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "reconciler" for t in node.targets)
            and isinstance(node.value, ast.Call)
            and _called_name(node.value) == "build_reconciler_for_run"
        ]
        assert builds, (
            f"{ensemble_dir.name} declares reconciliation={reconciliation!r} but its "
            f"main.py never assigns `reconciler = build_reconciler_for_run(...)` as code "
            f"— importing the name, or leaving the call in a comment, is not wiring it. "
            f"See EPIC #172 / ADR-014."
        )
        injections = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and any(
                kw.arg == "reconciler"
                and isinstance(kw.value, ast.Name)
                and kw.value.id == "reconciler"
                for kw in node.keywords
            )
        ]
        assert injections, (
            f"{ensemble_dir.name} builds a reconciler but no call passes "
            f"reconciler=reconciler to its manager — it will fail loud at runtime."
        )
        assert min(n.lineno for n in builds) < min(n.lineno for n in injections), (
            f"{ensemble_dir.name} passes reconciler=reconciler on line "
            f"{min(n.lineno for n in injections)} but only builds it on line "
            f"{min(n.lineno for n in builds)} — that is a use before assignment, and the "
            f"module raises before it reconciles anything."
        )

    def test_a_declared_reconciliation_is_one_pipeline_core_supports(self, ensemble_dir):
        """Separate from the wiring guard, because a typo'd mode and an unwired reconciler
        are different defects. The accepted set is IMPORTED from pipeline-core's sniffer
        rather than restated here, so this cannot drift from what the framework will
        actually accept at load — the same discipline #601 applied to the reconciliation
        geography registry after a hand-copied literal there proved wrong.
        """
        sniffer = pytest.importorskip(
            "views_pipeline_core.modules.validation.core_config_sniffer",
            reason="run_tests.yml installs pipeline-core; other jobs may not",
        )
        meta = load_config_module(
            ensemble_dir / "configs" / "config_meta.py"
        ).get_meta_config()
        reconciliation = meta.get("reconciliation")
        if not reconciliation:
            pytest.skip(f"{ensemble_dir.name} declares no reconciliation")
        supported = sniffer.SUPPORTED_RECONCILIATION_TYPES
        assert reconciliation in supported, (
            f"{ensemble_dir.name} declares reconciliation={reconciliation!r}, which "
            f"pipeline-core does not accept (supported: {sorted(supported)}). The config "
            f"would be refused at load."
        )

        # ⛔ Supported is not the same set as current. `SUPPORTED_RECONCILIATION_TYPES` is
        # `{'pgm_cm_point', 'pgm_cm'}` and `DEPRECATED_RECONCILIATION_TYPES` is
        # `{'pgm_cm_point'}`, so the assertion above catches a TYPO and would not catch a
        # regression: flipping `chrome_box` back to `pgm_cm_point` left the whole suite
        # green, and that mode — per the ensemble's own config_meta — "would collapse this
        # ensemble's 1,024 draws to a single number before reconciling".
        deprecated = getattr(sniffer, "DEPRECATED_RECONCILIATION_TYPES", None)
        assert deprecated is not None, (
            "pipeline-core's sniffer no longer exposes DEPRECATED_RECONCILIATION_TYPES, "
            "so this guard cannot see its subject and refuses rather than passing "
            "(ADR-025). Re-point it at wherever the deprecation now lives."
        )
        if ensemble_dir.name in RECONCILIATION_DEPRECATION_DEBT:
            assert reconciliation in deprecated, (
                f"{ensemble_dir.name} is listed in RECONCILIATION_DEPRECATION_DEBT but "
                f"declares {reconciliation!r}, which is not deprecated — if it has been "
                f"migrated, DELETE it from that set so the exemption stops standing."
            )
        else:
            assert reconciliation not in deprecated, (
                f"{ensemble_dir.name} declares reconciliation={reconciliation!r}, which "
                f"pipeline-core has DEPRECATED (deprecated: {sorted(deprecated)}). "
                f"`pgm_cm_point` reconciles a point: it collapses the posterior to one "
                f"number before reconciling, which for a distributional ensemble throws "
                f"away the forecast. Use `pgm_cm`, the frames-native draw-aware mode."
            )

    def test_declared_modelset_and_sample_counts_match_reality(self, ensemble_dir):
        """Opt-in belt-and-suspenders contract (ADR-015): an ensemble that declares
        ``expected_models`` / ``expected_samples_per_model`` in config_hyperparameters
        must have those numbers match reality —

          (a) expected_models == len(config_modelset["models"]), and
          (b) every constituent declares n_posterior_samples == expected_samples_per_model.

        PFE concat concatenates draws on the sample axis (pipeline-core
        prediction_frame_ensemble.py:99), so the pooled total is
        expected_models × expected_samples_per_model. Equal per-model counts give each
        constituent equal weight in the pooled mixture and a predictable pooled
        dimension — so a single declared number is the right contract, checked here at
        CI rather than discovered at run time. Ensembles that do not declare the fields
        are skipped (legacy-compatible).
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
                # The pooled sample axis is what each constituent actually EMITS.
                # For an ADR-067 family head that is D×K (n_posterior_samples ×
                # n_head_samples), not D alone (ADR-015 §6). get_produced_sample_count
                # reduces to n_posterior_samples for non-family models (K=1).
                n = get_produced_sample_count(model_dir)
                if n != expected_samples:
                    mismatches[model_name] = n
            assert not mismatches, (
                f"{ensemble_dir.name}: declares expected_samples_per_model="
                f"{expected_samples} but these constituents EMIT a different produced "
                f"count (n_posterior_samples × n_head_samples): {mismatches} — equal "
                f"counts keep each constituent equally weighted in the pooled mixture; "
                f"normalize them (ADR-015 §6)."
            )
