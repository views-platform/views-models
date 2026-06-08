from datetime import date


def _current_month_id() -> int:
    """VIEWS month_id for the current calendar month. Epoch: January 1980."""
    today = date.today()
    return (today.year - 1980) * 12 + today.month


def generate(steps: int = 36) -> dict:
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
