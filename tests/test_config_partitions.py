"""Tests that partition configs are consistent across all models, ensembles,
extractors, and postprocessors.

Each entity has its own self-contained config_partitions.py (required by the
framework's importlib-based loading). These tests verify that all use the
same canonical partition boundaries from meta/partitions.json.

Override mechanism (ADR-011): A config_partitions.py file may declare a
PARTITION_OVERRIDE comment to use non-standard values. Such files are
skipped with a warning rather than failing.
"""
import re
import warnings

import pytest

from tests.conftest import (
    ALL_PARTITION_DIRS, ALL_PARTITION_NAMES,
    load_canonical_partitions,
)

CANONICAL = load_canonical_partitions()

OVERRIDE_MARKER = "# PARTITION_OVERRIDE:"


def _extract_partition_tuples(source: str) -> dict:
    """Extract calibration/validation train/test tuples from source code."""
    result = {}
    for section in ("calibration", "validation"):
        section_match = re.search(
            rf'"{section}":\s*\{{([^}}]+)\}}', source, re.DOTALL
        )
        if section_match:
            block = section_match.group(1)
            for key in ("train", "test"):
                tuple_match = re.search(
                    rf'"{key}":\s*\((\d+),\s*(\d+)\)', block
                )
                if tuple_match:
                    result[f"{section}_{key}"] = (
                        int(tuple_match.group(1)),
                        int(tuple_match.group(2)),
                    )
    return result


def _has_override(source: str) -> bool:
    """Check if file declares a PARTITION_OVERRIDE."""
    return OVERRIDE_MARKER in source


def _read_partition_source(any_partition_dir):
    """Read config_partitions.py source from a directory."""
    return (any_partition_dir / "configs" / "config_partitions.py").read_text()


class TestPartitionConsistency:
    """Runs against all models, ensembles, extractors, and postprocessors."""

    @pytest.mark.parametrize(
        "any_partition_dir", ALL_PARTITION_DIRS, ids=ALL_PARTITION_NAMES
    )
    def test_partition_config_exists(self, any_partition_dir):
        assert (any_partition_dir / "configs" / "config_partitions.py").exists()

    @pytest.mark.parametrize(
        "any_partition_dir", ALL_PARTITION_DIRS, ids=ALL_PARTITION_NAMES
    )
    def test_has_generate_function(self, any_partition_dir):
        source = _read_partition_source(any_partition_dir)
        assert "def generate" in source, (
            f"{any_partition_dir.name} config_partitions.py has no generate() function"
        )

    @pytest.mark.parametrize(
        "any_partition_dir", ALL_PARTITION_DIRS, ids=ALL_PARTITION_NAMES
    )
    def test_calibration_train(self, any_partition_dir):
        source = _read_partition_source(any_partition_dir)
        if _has_override(source):
            pytest.skip(f"{any_partition_dir.name} has declared PARTITION_OVERRIDE")
        tuples = _extract_partition_tuples(source)
        expected = tuple(CANONICAL["calibration"]["train"])
        assert tuples.get("calibration_train") == expected, (
            f"{any_partition_dir.name} calibration train mismatch: "
            f"expected {expected}, found {tuples.get('calibration_train')}"
        )

    @pytest.mark.parametrize(
        "any_partition_dir", ALL_PARTITION_DIRS, ids=ALL_PARTITION_NAMES
    )
    def test_calibration_test(self, any_partition_dir):
        source = _read_partition_source(any_partition_dir)
        if _has_override(source):
            pytest.skip(f"{any_partition_dir.name} has declared PARTITION_OVERRIDE")
        tuples = _extract_partition_tuples(source)
        expected = tuple(CANONICAL["calibration"]["test"])
        assert tuples.get("calibration_test") == expected, (
            f"{any_partition_dir.name} calibration test mismatch: "
            f"expected {expected}, found {tuples.get('calibration_test')}"
        )

    @pytest.mark.parametrize(
        "any_partition_dir", ALL_PARTITION_DIRS, ids=ALL_PARTITION_NAMES
    )
    def test_validation_train(self, any_partition_dir):
        source = _read_partition_source(any_partition_dir)
        if _has_override(source):
            pytest.skip(f"{any_partition_dir.name} has declared PARTITION_OVERRIDE")
        tuples = _extract_partition_tuples(source)
        expected = tuple(CANONICAL["validation"]["train"])
        assert tuples.get("validation_train") == expected, (
            f"{any_partition_dir.name} validation train mismatch: "
            f"expected {expected}, found {tuples.get('validation_train')}"
        )

    @pytest.mark.parametrize(
        "any_partition_dir", ALL_PARTITION_DIRS, ids=ALL_PARTITION_NAMES
    )
    def test_validation_test(self, any_partition_dir):
        source = _read_partition_source(any_partition_dir)
        if _has_override(source):
            pytest.skip(f"{any_partition_dir.name} has declared PARTITION_OVERRIDE")
        tuples = _extract_partition_tuples(source)
        expected = tuple(CANONICAL["validation"]["test"])
        assert tuples.get("validation_test") == expected, (
            f"{any_partition_dir.name} validation test mismatch: "
            f"expected {expected}, found {tuples.get('validation_test')}"
        )

    @pytest.mark.parametrize(
        "any_partition_dir", ALL_PARTITION_DIRS, ids=ALL_PARTITION_NAMES
    )
    def test_forecasting_offset(self, any_partition_dir):
        """All entities should use ViewsMonth.now().id + forecasting_offset."""
        source = _read_partition_source(any_partition_dir)
        if _has_override(source):
            pytest.skip(f"{any_partition_dir.name} has declared PARTITION_OVERRIDE")
        expected_offset = str(abs(CANONICAL["forecasting_offset"]))
        offsets = re.findall(r'ViewsMonth\.now\(\)\.id\s*-\s*(\d+)', source)
        for offset in offsets:
            assert offset == expected_offset, (
                f"{any_partition_dir.name} uses forecasting offset -{offset} "
                f"instead of -{expected_offset}"
            )

    @pytest.mark.parametrize(
        "any_partition_dir", ALL_PARTITION_DIRS, ids=ALL_PARTITION_NAMES
    )
    def test_train_before_test(self, any_partition_dir):
        """Train end must be less than test start (no data leakage).

        Intentionally does NOT skip overrides — data leakage is a structural
        invariant that must hold regardless of boundary values.
        """
        source = _read_partition_source(any_partition_dir)
        tuples = _extract_partition_tuples(source)
        for section in ("calibration", "validation"):
            train = tuples.get(f"{section}_train")
            test = tuples.get(f"{section}_test")
            if train and test:
                assert train[1] < test[0], (
                    f"{any_partition_dir.name} {section}: train end ({train[1]}) "
                    f">= test start ({test[0]}) — potential data leakage"
                )


class TestPartitionOverrideDeclaration:
    """Ensure any non-standard partition file declares its override."""

    @pytest.mark.parametrize(
        "any_partition_dir", ALL_PARTITION_DIRS, ids=ALL_PARTITION_NAMES
    )
    def test_override_is_declared(self, any_partition_dir):
        """A file with non-canonical values MUST have a PARTITION_OVERRIDE comment."""
        source = _read_partition_source(any_partition_dir)
        tuples = _extract_partition_tuples(source)

        expected_tuples = {
            "calibration_train": tuple(CANONICAL["calibration"]["train"]),
            "calibration_test": tuple(CANONICAL["calibration"]["test"]),
            "validation_train": tuple(CANONICAL["validation"]["train"]),
            "validation_test": tuple(CANONICAL["validation"]["test"]),
        }

        deviations = []
        for key, expected in expected_tuples.items():
            actual = tuples.get(key)
            if actual and actual != expected:
                deviations.append(f"{key}: expected {expected}, found {actual}")

        if not deviations:
            return  # canonical — nothing to check

        if _has_override(source):
            warnings.warn(
                f"{any_partition_dir.name} uses non-standard partitions "
                f"(declared override): {'; '.join(deviations)}",
                stacklevel=1,
            )
            return  # declared override — warn but pass

        pytest.fail(
            f"CRITICAL: {any_partition_dir.name} uses non-standard partitions "
            f"without PARTITION_OVERRIDE declaration.\n"
            f"Deviations: {'; '.join(deviations)}\n"
            f"Either fix the values to match meta/partitions.json or add "
            f"'# PARTITION_OVERRIDE: <reason>' to declare the exception."
        )
