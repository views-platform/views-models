"""Partition domain rules and value types.

Contains the structural invariants, temporal plausibility checks,
and month_id encoding for VIEWS partition boundaries. No I/O.
"""
from dataclasses import dataclass
from datetime import date

TRAIN_START = 121
TEST_WINDOW = 48
MONTH_ID_EPOCH = 1980


def month_id_to_date(mid: int) -> str:
    year = MONTH_ID_EPOCH + (mid - 1) // 12
    month = ((mid - 1) % 12) + 1
    return f"{year}-{month:02d}"


def date_to_month_id(year: int, month: int) -> int:
    return (year - MONTH_ID_EPOCH) * 12 + month


def max_val_test_end() -> int:
    """Latest allowable validation test end: Dec of (current_year - 1).

    UCDP releases calibrated annual data covering the previous year.
    In 2026, the latest annual data covers through Dec 2025, so
    validation test cannot extend beyond month_id for Dec 2025.
    """
    return date_to_month_id(date.today().year - 1, 12)


@dataclass(frozen=True)
class PartitionBoundaries:
    """Immutable set of calibration and validation partition boundaries.

    Attributes use (start, end) tuples where both are inclusive month_ids.
    """

    cal_train: tuple[int, int]
    cal_test: tuple[int, int]
    val_train: tuple[int, int]
    val_test: tuple[int, int]

    def validate_invariants(self) -> list[str]:
        """Check 7 structural rules. Returns empty list if valid."""
        errors = []
        if self.cal_train[0] != TRAIN_START:
            errors.append(
                f"calibration train start must be {TRAIN_START}, "
                f"got {self.cal_train[0]}"
            )
        if self.val_train[0] != TRAIN_START:
            errors.append(
                f"validation train start must be {TRAIN_START}, "
                f"got {self.val_train[0]}"
            )
        if self.cal_test[0] != self.cal_train[1] + 1:
            errors.append(
                f"calibration test start ({self.cal_test[0]}) must be "
                f"calibration train end + 1 ({self.cal_train[1] + 1})"
            )
        if self.cal_test[1] - self.cal_test[0] + 1 != TEST_WINDOW:
            errors.append(
                f"calibration test window must be {TEST_WINDOW} months, "
                f"got {self.cal_test[1] - self.cal_test[0] + 1}"
            )
        if self.val_train[1] != self.cal_test[1]:
            errors.append(
                f"validation train end ({self.val_train[1]}) must equal "
                f"calibration test end ({self.cal_test[1]})"
            )
        if self.val_test[0] != self.val_train[1] + 1:
            errors.append(
                f"validation test start ({self.val_test[0]}) must be "
                f"validation train end + 1 ({self.val_train[1] + 1})"
            )
        if self.val_test[1] - self.val_test[0] + 1 != TEST_WINDOW:
            errors.append(
                f"validation test window must be {TEST_WINDOW} months, "
                f"got {self.val_test[1] - self.val_test[0] + 1}"
            )
        return errors

    def validate_temporal(self) -> list[str]:
        """Check that partitions don't extend beyond available UCDP data."""
        limit = max_val_test_end()
        if self.val_test[1] > limit:
            return [
                f"validation test end "
                f"({self.val_test[1]} = {month_id_to_date(self.val_test[1])}) "
                f"exceeds latest UCDP annual data "
                f"(Dec {date.today().year - 1} = month_id {limit}). "
                f"Use --force to override."
            ]
        return []

    def bumped(self, months: int) -> "PartitionBoundaries":
        """Return a new PartitionBoundaries advanced by N months."""
        return PartitionBoundaries(
            cal_train=(TRAIN_START, self.cal_train[1] + months),
            cal_test=(self.cal_test[0] + months, self.cal_test[1] + months),
            val_train=(TRAIN_START, self.val_train[1] + months),
            val_test=(self.val_test[0] + months, self.val_test[1] + months),
        )

    def to_flat_dict(self) -> dict[str, tuple[int, int]]:
        return {
            "calibration_train": self.cal_train,
            "calibration_test": self.cal_test,
            "validation_train": self.val_train,
            "validation_test": self.val_test,
        }

    def to_json_dict(self) -> dict:
        return {
            "calibration": {
                "train": list(self.cal_train),
                "test": list(self.cal_test),
            },
            "validation": {
                "train": list(self.val_train),
                "test": list(self.val_test),
            },
        }

    @classmethod
    def from_json(cls, data: dict) -> "PartitionBoundaries":
        return cls(
            cal_train=tuple(data["calibration"]["train"]),
            cal_test=tuple(data["calibration"]["test"]),
            val_train=tuple(data["validation"]["train"]),
            val_test=tuple(data["validation"]["test"]),
        )
