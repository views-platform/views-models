"""Partition definitions for shining_codex.

Defines temporal boundaries for each run type. These are identical
to all other VIEWS cm models — the partitions are a platform
convention, not model-specific.

    calibration:  train 121-444, test 445-492  (Jan 1990 – Dec 2020)
    validation:   train 121-492, test 493-540  (Jan 1990 – Dec 2024)
    forecasting:  train 121-now,  test now+1 to now+steps  (dynamic)

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
            "train": (121, 444),
            "test": (445, 492),
        },
        "validation": {
            "train": (121, 492),
            "test": (493, 540),
        },
        "forecasting": {
            "train": forecasting_train_range(),
            "test": forecasting_test_range(steps=steps),
        },
    }
