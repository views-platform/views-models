# Class Intent Contract: tools.partitions.domain.PartitionBoundaries

**Status:** Active
**Owner:** Project maintainers
**Last reviewed:** 2026-06-07
**Related ADRs:** ADR-011

---

## 1. Purpose

> `PartitionBoundaries` is an immutable value object representing the calibration and validation partition boundaries for VIEWS models. It encodes the 7 structural invariants, performs temporal plausibility checks against available UCDP data, and produces bumped copies for the annual partition advance. This is the domain layer — pure logic, no I/O.

Located in: `tools/partitions/domain.py`

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** read or write files (that's `fileops.py`'s job)
- Does **not** know about `config_partitions.py` file format
- Does **not** handle CLI arguments or user interaction (that's `bump.py`'s job)
- Does **not** manage the forecasting partition (which is dynamic and model-specific)

---

## 3. Responsibilities and Guarantees

- Stores 4 partition tuples: `cal_train`, `cal_test`, `val_train`, `val_test` — each `(start, end)` inclusive month_ids
- `validate_invariants()` checks 7 structural rules and returns a list of error strings (empty = valid):
  1. Calibration train start == 121 (Jan 1990)
  2. Validation train start == 121 (Jan 1990)
  3. Calibration test start == calibration train end + 1
  4. Calibration test window == 48 months
  5. Validation train end == calibration test end (chaining)
  6. Validation test start == validation train end + 1
  7. Validation test window == 48 months
- `validate_temporal()` checks that validation test end does not exceed Dec (current_year - 1)
- `bumped(months)` returns a new `PartitionBoundaries` advanced by N months, preserving all invariants
- `from_json(data)` constructs from the `meta/partitions.json` format
- `to_json_dict()` and `to_flat_dict()` produce serializable representations
- `month_id_to_date()` and `date_to_month_id()` convert between month_ids and `YYYY-MM` strings
- `max_val_test_end()` returns the temporal plausibility limit as a month_id

---

## 4. Inputs and Assumptions

- Assumes month_id epoch is 1980 (`MONTH_ID_EPOCH = 1980`)
- Assumes all month_id values are positive integers
- Assumes `date.today()` returns the current date (used by `validate_temporal` and `max_val_test_end`)
- `from_json()` assumes the input dict has `calibration.train`, `calibration.test`, `validation.train`, `validation.test` keys, each containing 2-element lists

---

## 5. Outputs and Side Effects

- All methods are pure — no side effects, no I/O, no logging
- `validate_invariants()` and `validate_temporal()` return `list[str]` (empty = valid, non-empty = error messages)
- `bumped()` returns a new frozen dataclass instance — the original is never modified
- `to_flat_dict()` returns `dict[str, tuple[int, int]]` with keys like `"calibration_train"`
- `to_json_dict()` returns nested dict matching `meta/partitions.json` structure

---

## 6. Failure Modes and Loudness

- `from_json()` raises `KeyError` if required keys are missing — caller must catch
- `from_json()` raises `TypeError` if values are not iterable — caller must catch
- `validate_invariants()` never raises — returns error strings
- `validate_temporal()` never raises — returns error strings
- `bumped(0)` is the identity — returns an equal copy
- `bumped(negative)` produces values that will fail `validate_invariants()` — no explicit guard

---

## 7. Boundaries and Interactions

- Depends on: `dataclasses`, `datetime.date` (stdlib only)
- Called by: `tools/partitions/bump.py` (loads canonical, validates, bumps, validates again)
- Called by: `tests/test_bump_partitions.py` (unit tests for all methods)
- Does NOT interact with: `fileops.py`, `config_partitions.py` files, `meta/partitions.json`

---

## 8. Examples of Correct Usage

```python
from tools.partitions.domain import PartitionBoundaries

boundaries = PartitionBoundaries(
    cal_train=(121, 444), cal_test=(445, 492),
    val_train=(121, 492), val_test=(493, 540),
)
assert boundaries.validate_invariants() == []

bumped = boundaries.bumped(12)
assert bumped.validate_invariants() == []
assert bumped.val_test == (505, 552)
```

---

## 9. Examples of Incorrect Usage

```python
# Wrong: constructing with values that violate invariants
bad = PartitionBoundaries(
    cal_train=(100, 444), cal_test=(445, 492),
    val_train=(121, 492), val_test=(493, 540),
)
# No error at construction — must call validate_invariants() to detect

# Wrong: assuming bumped() validates
bumped = boundaries.bumped(1200)
# Returns successfully — caller must validate_temporal() to detect the problem
```

---

## 10. Test Alignment

- `tests/test_bump_partitions.py::TestMonthIdConversion` — 5 tests for month_id encoding
- `tests/test_bump_partitions.py::TestPartitionBoundariesInvariants` — 5 tests for invariant validation
- `tests/test_bump_partitions.py::TestTemporalPlausibility` — 4 tests including double-bump and absurd-bump blocking
- `tests/test_bump_partitions.py::TestBumpedValues` — 4 tests for bump arithmetic
- `tests/test_falsify_bump_robustness.py` — 3 verification tests for resolved findings

---

## 11. Evolution Notes

- `bumped()` could validate before returning, but currently relies on the caller to validate. This is deliberate — separation of computation and validation.
- If the test window size changes from 48 months, `TEST_WINDOW` must be updated here AND in `meta/partitions.json` documentation.
- `validate_temporal()` uses `date.today()` which makes it impure in the strictest sense. Injection via parameter would improve testability for time-sensitive edge cases.

---

## End of Contract

This document defines the **intended meaning** of `tools.partitions.domain.PartitionBoundaries`.

Changes to behavior that violate this intent are bugs.
Changes to intent must update this contract.
