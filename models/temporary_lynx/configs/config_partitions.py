"""Partition definitions for temporary_lynx.

Defines temporal boundaries for each run type. These are identical
to all other VIEWS pgm models — the partitions are a platform
convention, not model-specific.

    See ``meta/partitions.json`` for the canonical calibration/validation
    train/test ranges (rewritten across all models by the partition bump
    tool); forecasting is dynamic from the current month.

Month IDs use VIEWS encoding: month_id = (year - 1980) * 12 + month.
"""

from datetime import date


def _current_month_id() -> int:
    """VIEWS month_id for the current calendar month. Epoch: January 1980."""
    today = date.today()
    return (today.year - 1980) * 12 + today.month


def generate(steps: int = 36) -> dict:
    """Return partition dict with train/test month_id ranges.

    Args:
        steps: Forecast horizon in months (default 36 = 3 years).

    Returns:
        Dict with keys "calibration", "validation", "forecasting",
        each containing {"train": (start, end), "test": (start, end)}.
    """

    def forecasting_train_range():
        return (121, _current_month_id() - 1)

    def forecasting_test_range(steps):
        month_last = _current_month_id() - 1
        return (month_last + 1, month_last + 1 + steps)

    return {
        "calibration": {
            "train": (121, 456),
            "test": (457, 504),
        },
        "validation": {
            "train": (121, 504),
            "test": (505, 552),
        },
        "forecasting": {
            "train": forecasting_train_range(),
            "test": forecasting_test_range(steps=steps),
        },
    }
