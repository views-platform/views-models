"""Regression-target naming contracts — config is the single source of truth (EPIC #154).

These are **agnostic** contracts: they assert *structure* (a model's targets are
declared and discoverable via the one accessor, and the two declaration locations
agree) without ever referencing a literal target name. A model may call its target
whatever it likes; nothing here hardcodes `lr_ged_sb` or any other string.

The accessor under test (`tests/conftest.py::get_regression_targets`) is THE way
views-models code should obtain a model's targets — see S2–S6 for its adoption
across the test suite and tooling.
"""
import pytest

from tests.conftest import (
    ALL_MODEL_DIRS,
    MODEL_NAMES,
    get_regression_targets,
    regression_targets_by_location,
)

pytestmark = pytest.mark.beige


@pytest.mark.parametrize("model_dir", ALL_MODEL_DIRS, ids=MODEL_NAMES)
def test_meta_and_hp_agree(model_dir):
    """If a model declares regression_targets in BOTH config_meta and
    config_hyperparameters, the two must agree. (Single-location is fine.)"""
    located = regression_targets_by_location(model_dir)
    if "meta" in located and "hp" in located:
        assert sorted(located["meta"]) == sorted(located["hp"]), (
            f"{model_dir.name}: config_meta regression_targets {located['meta']} != "
            f"config_hyperparameters {located['hp']} — the two declarations must agree"
        )


@pytest.mark.parametrize("model_dir", ALL_MODEL_DIRS, ids=MODEL_NAMES)
def test_accessor_resolves_declared_targets(model_dir):
    """The accessor resolves a model's declared targets (config_meta precedence,
    config_hyperparameters fallback) — and resolves nothing only when nothing is
    declared. Dogfoods the single source of truth across all archetypes."""
    located = regression_targets_by_location(model_dir)
    resolved = get_regression_targets(model_dir)
    if located:
        expected = located.get("meta") or located.get("hp")
        assert resolved == expected, (
            f"{model_dir.name}: accessor returned {resolved} but expected {expected} "
            f"(meta precedence over hp) from {located}"
        )
        assert resolved, f"{model_dir.name}: declares targets {located} but accessor resolved none"
    else:
        assert resolved == [], (
            f"{model_dir.name}: declares no regression_targets but accessor returned {resolved}"
        )
