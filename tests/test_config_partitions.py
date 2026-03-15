"""Tests that partition configs are consistent across all models and ensembles.

Each model/ensemble has its own self-contained config_partitions.py (required
by the framework's importlib-based loading). These tests verify that all use
the same canonical partition boundaries and forecasting offset.
"""
import re


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


class TestPartitionConsistency:
    """Runs against all models AND ensembles via any_model_dir fixture."""

    def test_partition_config_exists(self, any_model_dir):
        assert (any_model_dir / "configs" / "config_partitions.py").exists()

    def test_has_generate_function(self, any_model_dir):
        source = (any_model_dir / "configs" / "config_partitions.py").read_text()
        assert "def generate" in source, (
            f"{any_model_dir.name} config_partitions.py has no generate() function"
        )

    def test_calibration_train(self, any_model_dir):
        source = (any_model_dir / "configs" / "config_partitions.py").read_text()
        tuples = _extract_partition_tuples(source)
        assert tuples.get("calibration_train") == (121, 444), (
            f"{any_model_dir.name} calibration train mismatch: "
            f"{tuples.get('calibration_train')}"
        )

    def test_calibration_test(self, any_model_dir):
        source = (any_model_dir / "configs" / "config_partitions.py").read_text()
        tuples = _extract_partition_tuples(source)
        assert tuples.get("calibration_test") == (445, 492), (
            f"{any_model_dir.name} calibration test mismatch: "
            f"{tuples.get('calibration_test')}"
        )

    def test_validation_train(self, any_model_dir):
        source = (any_model_dir / "configs" / "config_partitions.py").read_text()
        tuples = _extract_partition_tuples(source)
        assert tuples.get("validation_train") == (121, 492), (
            f"{any_model_dir.name} validation train mismatch: "
            f"{tuples.get('validation_train')}"
        )

    def test_validation_test(self, any_model_dir):
        source = (any_model_dir / "configs" / "config_partitions.py").read_text()
        tuples = _extract_partition_tuples(source)
        assert tuples.get("validation_test") == (493, 540), (
            f"{any_model_dir.name} validation test mismatch: "
            f"{tuples.get('validation_test')}"
        )

    def test_forecasting_offset_is_minus_one(self, any_model_dir):
        """All models/ensembles should use ViewsMonth.now().id - 1."""
        source = (any_model_dir / "configs" / "config_partitions.py").read_text()
        offsets = re.findall(r'ViewsMonth\.now\(\)\.id\s*-\s*(\d+)', source)
        for offset in offsets:
            assert offset == "1", (
                f"{any_model_dir.name} uses forecasting offset -{offset} "
                f"instead of -1"
            )
