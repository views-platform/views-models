def generate(steps: int = 36) -> dict:
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
            "train": (121, 540),
            "test": (541, 541 + steps),
        },
    }
