"""Target names carry the prefix their kind requires (ADR-012).

ADR-012 fixes two prefixes: ``lr_`` marks a **regression** target on its original
measurement scale, ``by_`` marks a **classification** target derived from counts. All
other prefixes (``ln_``, ``lx_``, …) are deprecated and must not appear as targets in
new configs.

The ADR proposed this guard itself — *"Existing tests (`test_config_completeness.py`)
can be extended to assert that all `regression_targets` use the `lr_` prefix and all
`classification_targets` use the `by_` prefix"* — and it was never written. It lives in
its own file rather than inside `test_config_completeness.py` because it is a different
question: that file asks whether required keys are *present*, this one asks whether the
values are *well-formed*.

**Why now.** views-models#367 declared `classification_targets` without the metric key
that obliges, and nothing caught it. That is the same family: a targets declaration that
no test inspected. #374 closed the metric half by loading every config through
pipeline-core's `CoreConfigSniffer`; this closes the prefix half, which the sniffer does
not check — it validates the targets↔metrics *pairing*, never the target *names*.

**What this does not do.** It asserts the prefix and nothing about the suffix, the scale,
or whether the target exists in the data. ADR-012 is explicit that the prefix is an
identity convention: `lr_` does not mean a transform was applied or needs undoing.
Queryset-level transforms are `tools/audit/queryset_transforms.py`'s job.
"""

import json
from pathlib import Path

import pytest

from tests.conftest import ALL_ENSEMBLE_DIRS, ALL_MODEL_DIRS, load_config_module

pytestmark = [pytest.mark.beige]

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The canonical registry of not-real entities. Loaded, never hardcoded — the same rule
#: `tools/partitions/fileops.py` and `tools/catalogs/create_catalogs.py` follow, pinned
#: by `test_bump_partitions.py::TestFixtureSetConsistency` (C-61, #99).
FIXTURES_PATH = REPO_ROOT / "meta" / "fixtures.json"

REGRESSION_PREFIX = "lr_"
CLASSIFICATION_PREFIX = "by_"

#: Both config files that may declare targets. Models put them in hyperparameters,
#: ensembles in meta; several declare in both, so each is read wherever it appears.
TARGET_SOURCES = (
    ("config_hyperparameters.py", "get_hp_config"),
    ("config_meta.py", "get_meta_config"),
)


def _fixture_names():
    return set(json.loads(FIXTURES_PATH.read_text()))


def _declared_targets(directory):
    """{"regression_targets": [...], "classification_targets": [...]} across both files."""
    found = {"regression_targets": [], "classification_targets": []}
    for filename, getter in TARGET_SOURCES:
        path = directory / "configs" / filename
        if not path.exists():
            continue
        fn = getattr(load_config_module(path), getter, None)
        if fn is None:
            continue
        config = fn() or {}
        for key in found:
            found[key].extend(config.get(key) or [])
    return found


def _real_sources():
    """Every non-fixture model and ensemble directory.

    Fixtures are excluded because their targets are deliberately synthetic
    (`synth_target`) — nine of them declare it, and every one is registered in
    `meta/fixtures.json`. Excluding by that registry rather than by prefix keeps the
    exclusion a declared fact rather than a circular one: a real model that started
    using `synth_` would still fail.
    """
    fixtures = _fixture_names()
    for directory in list(ALL_MODEL_DIRS) + list(ALL_ENSEMBLE_DIRS):
        if directory.name in fixtures:
            continue
        yield directory


def _offenders(key, prefix):
    bad = {}
    for directory in _real_sources():
        wrong = sorted(
            {t for t in _declared_targets(directory)[key] if not t.startswith(prefix)}
        )
        if wrong:
            bad[directory.name] = wrong
    return bad


def test_regression_targets_use_the_lr_prefix():
    """ADR-012: `lr_` marks a target on its original measurement scale."""
    offenders = _offenders("regression_targets", REGRESSION_PREFIX)
    assert not offenders, (
        f"regression_targets must use the '{REGRESSION_PREFIX}' prefix (ADR-012):\n"
        + "\n".join(f"  {name}: {targets}" for name, targets in sorted(offenders.items()))
    )


def test_classification_targets_use_the_by_prefix():
    """ADR-012: `by_` marks a classification target derived from counts."""
    offenders = _offenders("classification_targets", CLASSIFICATION_PREFIX)
    assert not offenders, (
        f"classification_targets must use the '{CLASSIFICATION_PREFIX}' prefix (ADR-012):\n"
        + "\n".join(f"  {name}: {targets}" for name, targets in sorted(offenders.items()))
    )


def test_the_prefix_checks_are_not_vacuous():
    """Both assertions must be inspecting real targets, not an empty set.

    If discovery or config loading broke, the two tests above would pass while reading
    nothing. Floors are set well under today's counts (190 regression, 24 classification
    across non-fixture sources) so ordinary churn does not trip them.
    """
    counts = {"regression_targets": 0, "classification_targets": 0}
    for directory in _real_sources():
        declared = _declared_targets(directory)
        for key in counts:
            counts[key] += len(declared[key])

    assert counts["regression_targets"] > 100, (
        f"only {counts['regression_targets']} regression targets seen — the prefix "
        f"assertion is passing over almost nothing"
    )
    assert counts["classification_targets"] > 10, (
        f"only {counts['classification_targets']} classification targets seen — the "
        f"prefix assertion is passing over almost nothing"
    )


def test_fixtures_are_excluded_by_the_registry_not_by_the_prefix():
    """The exclusion is a declared fact, and it is load-bearing.

    Nine fixture entities declare `synth_target` as a regression target. If they were
    not registered in `meta/fixtures.json`, the regression assertion would fail — so
    this pins that the registry is what excludes them, and that it still contains them.
    """
    fixtures = _fixture_names()
    synthetic = {
        directory.name
        for directory in list(ALL_MODEL_DIRS) + list(ALL_ENSEMBLE_DIRS)
        if any(
            not t.startswith(REGRESSION_PREFIX)
            for t in _declared_targets(directory)["regression_targets"]
        )
    }
    assert synthetic, "no synthetic-target entity found — this test has lost its subject"
    assert synthetic <= fixtures, (
        f"these declare non-{REGRESSION_PREFIX} regression targets but are NOT in "
        f"meta/fixtures.json: {sorted(synthetic - fixtures)}. Either register them as "
        f"fixtures or give them ADR-012 target names."
    )
