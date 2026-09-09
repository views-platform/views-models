def generate(steps: int = 36) -> dict:
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
            "train": (121, 540),
            "test": (541, 541 + steps),
        },
    }
