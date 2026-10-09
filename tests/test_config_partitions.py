"""Tests that partition configs are consistent across all models, ensembles,
extractors, and postprocessors.

Each entity has its own self-contained config_partitions.py (required by the
framework's importlib-based loading). These tests verify that all use the
same canonical partition boundaries from meta/partitions.json.
"""
import re

import pytest

from tests.conftest import (
    ALL_PARTITION_DIRS, ALL_PARTITION_NAMES,
    load_canonical_partitions,
)
from tools.partitions.fileops import extract_values as _extract_partition_tuples

pytestmark = pytest.mark.green

CANONICAL = load_canonical_partitions()


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
        tuples = _extract_partition_tuples(source)
        expected = tuple(CANONICAL["validation"]["test"])
        assert tuples.get("validation_test") == expected, (
            f"{any_partition_dir.name} validation test mismatch: "
            f"expected {expected}, found {tuples.get('validation_test')}"
        )

    @pytest.mark.parametrize(
        "any_partition_dir", ALL_PARTITION_DIRS, ids=ALL_PARTITION_NAMES
    )
    def test_forecasting_uses_current_month_id(self, any_partition_dir):
        """Dynamic partition files should use _current_month_id(), not ViewsMonth."""
        source = _read_partition_source(any_partition_dir)
        if "_current_month_id" not in source:
            pytest.skip(f"{any_partition_dir.name} uses static forecasting")
        assert "ViewsMonth" not in source, (
            f"{any_partition_dir.name} still imports ViewsMonth — "
            f"should use _current_month_id() instead"
        )
        offsets = re.findall(r'_current_month_id\(\)\s*-\s*(\d+)', source)
        for offset in offsets:
            assert offset == "1", (
                f"{any_partition_dir.name} uses forecasting offset -{offset} "
                f"instead of -1"
            )

    @pytest.mark.parametrize(
        "any_partition_dir", ALL_PARTITION_DIRS, ids=ALL_PARTITION_NAMES
    )
    def test_no_ingester3_dependency(self, any_partition_dir):
        """No partition config should depend on ingester3."""
        source = _read_partition_source(any_partition_dir)
        assert "ingester3" not in source, (
            f"{any_partition_dir.name} still imports from ingester3 — "
            f"use _current_month_id() with datetime.date instead"
        )

    @pytest.mark.parametrize(
        "any_partition_dir", ALL_PARTITION_DIRS, ids=ALL_PARTITION_NAMES
    )
    def test_train_before_test(self, any_partition_dir):
        """Train end must be less than test start (no data leakage)."""
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
