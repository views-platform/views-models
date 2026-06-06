from datetime import date


def _current_month_id() -> int:
    """VIEWS month_id for the current calendar month. Epoch: January 1980."""
    today = date.today()
    return (today.year - 1980) * 12 + today.month



def generate(steps: int = 36) -> dict:
    """
    Generates partition configurations for different phases of model evaluation.

    Returns:
        dict: A dictionary with keys 'calibration', 'validation', and 'forecasting', each containing
            'train' and 'test' tuples or callables specifying the index ranges for training and testing data.

    Partition details:
        - 'calibration': Uses fixed index ranges for training and testing.
        - 'validation': Uses fixed index ranges for training and testing.
        - 'forecasting': Uses the current month to dynamically determine training and testing index ranges.

    Note:
        - The 'forecasting' partition's 'train' and 'test' ranges are computed from the current month.
    """

    def forecasting_train_range():
        month_last = _current_month_id() - 1
        return (121, month_last)

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