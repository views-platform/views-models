"""Verification tests: partition bump robustness findings are resolved.

Original findings (falsification audit, 2026-06-06):
- P1: Regex parser failed on single-quoted Python files
- P3: No guard against double-bump (running --execute twice)
- P5: No temporal plausibility check (--bump 1200 passed all checks)

Resolution: tools/partitions/ package with PartitionBoundaries.validate_temporal()
and quote-resilient regex in fileops.extract_values().
"""
from tools.partitions.domain import PartitionBoundaries
from tools.partitions.fileops import extract_values

CURRENT = PartitionBoundaries(
    cal_train=(121, 444),
    cal_test=(445, 492),
    val_train=(121, 492),
    val_test=(493, 540),
)


class TestP3_DoubleBumpBlocked:
    """Double-bump is blocked by temporal plausibility: bumping once
    lands at Dec 2025 (the limit for 2026); bumping again exceeds it."""

    def test_second_bump_fails_temporal(self):
        once = CURRENT.bumped(12)
        assert once.validate_temporal() == []
        twice = once.bumped(12)
        errors = twice.validate_temporal()
        assert len(errors) == 1
        assert "exceeds" in errors[0]


class TestP5_AbsurdBumpBlocked:
    """Absurd bumps are blocked by temporal plausibility."""

    def test_bump_1200_rejected(self):
        absurd = CURRENT.bumped(1200)
        errors = absurd.validate_temporal()
        assert len(errors) == 1
        assert "2124" in errors[0]


class TestP1_SingleQuoteResilience:
    """The parser now handles both single and double quoted files."""

    def test_single_quotes_parse(self):
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
