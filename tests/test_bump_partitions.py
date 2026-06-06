"""Tests for the partition bump tooling.

Covers domain invariants, temporal plausibility, file parsing
(including single-quote resilience), round-trip rewriting, and
double-bump detection.
"""
import hashlib
from pathlib import Path

import pytest

from tools.partitions.domain import (
    PartitionBoundaries,
    date_to_month_id,
    max_val_test_end,
    month_id_to_date,
)
from tools.partitions.fileops import extract_values, rewrite_values

REPO_ROOT = Path(__file__).resolve().parent.parent

CURRENT = PartitionBoundaries(
    cal_train=(121, 444),
    cal_test=(445, 492),
    val_train=(121, 492),
    val_test=(493, 540),
)


class TestMonthIdConversion:
    def test_epoch(self):
        assert date_to_month_id(1980, 1) == 1

    def test_known_value(self):
        assert date_to_month_id(1990, 1) == 121

    def test_round_trip(self):
        assert month_id_to_date(121) == "1990-01"
        assert month_id_to_date(540) == "2024-12"
        assert month_id_to_date(552) == "2025-12"

    def test_dec_2016(self):
        assert date_to_month_id(2016, 12) == 444

    def test_dec_2020(self):
        assert date_to_month_id(2020, 12) == 492


class TestPartitionBoundariesInvariants:
    def test_current_values_are_valid(self):
        assert CURRENT.validate_invariants() == []

    def test_bumped_values_are_valid(self):
        bumped = CURRENT.bumped(12)
        assert bumped.validate_invariants() == []

    def test_wrong_train_start_rejected(self):
        bad = PartitionBoundaries(
            cal_train=(100, 444),
            cal_test=(445, 492),
            val_train=(121, 492),
            val_test=(493, 540),
        )
        errors = bad.validate_invariants()
        assert len(errors) >= 1
        assert "121" in errors[0]

    def test_broken_chain_rejected(self):
        bad = PartitionBoundaries(
            cal_train=(121, 444),
            cal_test=(445, 492),
            val_train=(121, 500),
            val_test=(501, 548),
        )
        errors = bad.validate_invariants()
        assert any("must equal" in e for e in errors)

    def test_wrong_window_rejected(self):
        bad = PartitionBoundaries(
            cal_train=(121, 444),
            cal_test=(445, 480),
            val_train=(121, 480),
            val_test=(481, 528),
        )
        errors = bad.validate_invariants()
        assert any("48 months" in e for e in errors)


class TestTemporalPlausibility:
    def test_normal_bump_passes(self):
        bumped = CURRENT.bumped(12)
        assert bumped.validate_temporal() == []

    def test_double_bump_blocked(self):
        once = CURRENT.bumped(12)
        twice = once.bumped(12)
        errors = twice.validate_temporal()
        assert len(errors) == 1
        assert "exceeds" in errors[0]

    def test_absurd_bump_blocked(self):
        absurd = CURRENT.bumped(1200)
        errors = absurd.validate_temporal()
        assert len(errors) == 1
        assert "2124" in errors[0]

    def test_max_val_test_end_is_dec_previous_year(self):
        from datetime import date
        limit = max_val_test_end()
        expected = date_to_month_id(date.today().year - 1, 12)
        assert limit == expected


class TestBumpedValues:
    def test_train_start_never_moves(self):
        bumped = CURRENT.bumped(12)
        assert bumped.cal_train[0] == 121
        assert bumped.val_train[0] == 121

    def test_endpoints_advance_by_bump(self):
        bumped = CURRENT.bumped(12)
        assert bumped.cal_train[1] == 444 + 12
        assert bumped.cal_test == (445 + 12, 492 + 12)
        assert bumped.val_train[1] == 492 + 12
        assert bumped.val_test == (493 + 12, 540 + 12)

    def test_chain_invariant_preserved(self):
        bumped = CURRENT.bumped(12)
        assert bumped.val_train[1] == bumped.cal_test[1]
        assert bumped.val_test[0] == bumped.val_train[1] + 1

    def test_zero_bump_is_identity(self):
        same = CURRENT.bumped(0)
        assert same == CURRENT


class TestExtractValues:
    def test_double_quoted_file(self):
        source = '''
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
    }
'''
        result = extract_values(source)
        assert result is not None
        assert result["calibration_train"] == (121, 444)
        assert result["validation_test"] == (493, 540)

    def test_single_quoted_file(self):
        source = """
def generate(steps: int = 36) -> dict:
    return {
        'calibration': {
            'train': (121, 444),
            'test': (445, 492),
        },
        'validation': {
            'train': (121, 492),
            'test': (493, 540),
        },
    }
"""
        result = extract_values(source)
        assert result is not None
        assert result["calibration_train"] == (121, 444)
        assert result["validation_test"] == (493, 540)

    def test_all_repo_variants_parse(self):
        """Every unique config_partitions.py variant in the repo must parse."""
        files = sorted(
            list(REPO_ROOT.glob("models/*/configs/config_partitions.py"))
            + list(REPO_ROOT.glob("ensembles/*/configs/config_partitions.py"))
            + list(REPO_ROOT.glob("extractors/*/configs/config_partitions.py"))
            + list(REPO_ROOT.glob("postprocessors/*/configs/config_partitions.py"))
        )
        seen_hashes = set()
        for f in files:
            content = f.read_text()
            h = hashlib.md5(content.encode()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            result = extract_values(content)
            assert result is not None, (
                f"Failed to parse {f.relative_to(REPO_ROOT)}"
            )
            assert result["calibration_train"] == (121, 444)


class TestRewriteRoundTrip:
    _TEMPLATE = '''
def generate(steps: int = 36) -> dict:
    return {{
        {q}calibration{q}: {{
            {q}train{q}: (121, 444),
            {q}test{q}: (445, 492),
        }},
        {q}validation{q}: {{
            {q}train{q}: (121, 492),
            {q}test{q}: (493, 540),
        }},
        {q}forecasting{q}: {{
            {q}train{q}: (121, 540),
            {q}test{q}: (541, 541 + steps),
        }},
    }}
'''

    @pytest.mark.parametrize("quote", ['"', "'"], ids=["double", "single"])
    def test_round_trip(self, quote):
        source = self._TEMPLATE.format(q=quote)
        new_vals = {
            "calibration_train": (121, 456),
            "calibration_test": (457, 504),
            "validation_train": (121, 504),
            "validation_test": (505, 552),
        }
        rewritten = rewrite_values(source, new_vals)
        extracted = extract_values(rewritten)
        assert extracted == new_vals

    @pytest.mark.parametrize("quote", ['"', "'"], ids=["double", "single"])
    def test_forecasting_untouched(self, quote):
        source = self._TEMPLATE.format(q=quote)
        new_vals = CURRENT.bumped(12).to_flat_dict()
        rewritten = rewrite_values(source, new_vals)
        assert "(121, 540)" in rewritten
        assert "(541, 541 + steps)" in rewritten
