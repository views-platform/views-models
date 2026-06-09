"""Tests for the partition bump tooling.

Covers domain invariants, temporal plausibility, file parsing
(including single-quote resilience), round-trip rewriting, and
double-bump detection.
"""
import hashlib
import json as _json
import sys as _sys
from pathlib import Path

import pytest

from tools.partitions.bump import main as bump_main

from tools.partitions.domain import (
    PartitionBoundaries,
    date_to_month_id,
    max_val_test_end,
    month_id_to_date,
)
from tools.partitions.fileops import (
    discover_entity_dirs,
    discover_partition_files,
    extract_values,
    has_partition_override,
    rewrite_values,
)


pytestmark = pytest.mark.green
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


class TestDiscoverEntityDirs:
    def test_finds_models_with_main_py(self, tmp_path):
        models = tmp_path / "models"
        (models / "real_model" / "configs").mkdir(parents=True)
        (models / "real_model" / "main.py").touch()
        (models / "scaffold_only" / "configs").mkdir(parents=True)
        result = discover_entity_dirs(tmp_path)
        names = [d.name for d in result]
        assert "real_model" in names
        assert "scaffold_only" not in names

    def test_excludes_fixtures(self, tmp_path):
        models = tmp_path / "models"
        (models / "fake_model").mkdir(parents=True)
        (models / "fake_model" / "main.py").touch()
        (models / "test_model").mkdir(parents=True)
        (models / "test_model" / "main.py").touch()
        result = discover_entity_dirs(tmp_path)
        names = [d.name for d in result]
        assert "fake_model" not in names
        assert "test_model" not in names

    def test_includes_extractors_and_postprocessors(self, tmp_path):
        ext = tmp_path / "extractors" / "my_extractor" / "configs"
        ext.mkdir(parents=True)
        (ext / "config_partitions.py").touch()
        result = discover_entity_dirs(tmp_path)
        names = [d.name for d in result]
        assert "my_extractor" in names

    def test_coverage_matches_partition_files(self):
        """Every real entity in the repo should have a partition file."""
        entities = discover_entity_dirs(REPO_ROOT)
        partition_files = discover_partition_files(REPO_ROOT)
        partition_parents = {f.parent.parent for f in partition_files}
        missing = [d for d in entities if d not in partition_parents]
        assert len(missing) == 0, (
            f"Entities missing partition files: {[d.name for d in missing]}"
        )


class TestPartitionOverrideFlag:
    def test_detects_override_true(self):
        source = "PARTITION_OVERRIDE = True\n\ndef generate(): pass"
        assert has_partition_override(source) is True

    def test_ignores_override_false(self):
        source = "PARTITION_OVERRIDE = False\n\ndef generate(): pass"
        assert has_partition_override(source) is False

    def test_no_flag_means_no_override(self):
        source = "def generate(): pass"
        assert has_partition_override(source) is False

    def test_comment_does_not_count(self):
        source = "# PARTITION_OVERRIDE = True\n\ndef generate(): pass"
        assert has_partition_override(source) is False

    def test_real_repo_has_no_overrides_currently(self):
        """No production model currently uses PARTITION_OVERRIDE = True."""
        files = discover_partition_files(REPO_ROOT)
        overrides = []
        for f in files:
            if has_partition_override(f.read_text()):
                overrides.append(f.relative_to(REPO_ROOT))
        assert len(overrides) == 0, (
            f"Unexpected override files: {overrides}"
        )


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


class TestFixtureSetConsistency:
    """All fixture exclusion lists must reference the same canonical set."""

    def test_all_fixture_lists_match_canonical(self):
        import json
        canonical = set(json.load(open(REPO_ROOT / "meta" / "fixtures.json")))

        # tools/partitions/fileops.py
        from tools.partitions.fileops import _FIXTURE_NAMES
        assert _FIXTURE_NAMES == canonical, (
            f"fileops._FIXTURE_NAMES diverges from meta/fixtures.json: "
            f"extra={_FIXTURE_NAMES - canonical}, missing={canonical - _FIXTURE_NAMES}"
        )

        # tools/catalogs/create_catalogs.py — read via AST to avoid pipeline-core import
        import ast
        source = (REPO_ROOT / "tools" / "catalogs" / "create_catalogs.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "_FIXTURE_ENTRIES":
                        catalog_fixtures = {elt.value for elt in node.value.elts}
                        assert catalog_fixtures == canonical, (
                            f"create_catalogs._FIXTURE_ENTRIES diverges: "
                            f"extra={catalog_fixtures - canonical}, "
                            f"missing={canonical - catalog_fixtures}"
                        )


# ---------------------------------------------------------------------------
# Red tests: adversarial inputs and error paths
# ---------------------------------------------------------------------------

@pytest.mark.red
class TestAdversarialInputs:
    """Error paths and adversarial inputs for partition tooling."""

    def test_extract_values_garbage_input(self):
        assert extract_values("not python at all }{{{") is None

    def test_extract_values_partial_structure(self):
        source = '"calibration": {"train": (121, 444)}'
        assert extract_values(source) is None

    def test_extract_values_negative_month_ids_rejected(self):
        """Regex uses \\d+ which correctly rejects negative integers."""
        source = '''
def generate():
    return {
        "calibration": {"train": (-1, -1), "test": (-1, -1)},
        "validation": {"train": (-1, -1), "test": (-1, -1)},
    }
'''
        assert extract_values(source) is None

    def test_rewrite_values_no_return_statement(self):
        with pytest.raises(ValueError, match="return"):
            rewrite_values("x = 1", CURRENT.to_flat_dict())

    def test_rewrite_values_missing_section(self):
        source = 'return {"calibration": {"train": (1, 2), "test": (3, 4)}}'
        with pytest.raises(ValueError, match="validation"):
            rewrite_values(source, CURRENT.to_flat_dict())

    def test_bumped_negative_is_structurally_valid(self):
        """Negative bump goes backward — structurally valid, temporally valid
        (within available data). No guard against this — caller must check."""
        bumped = CURRENT.bumped(-12)
        assert bumped.validate_invariants() == []
        assert bumped.val_test[1] < CURRENT.val_test[1]

    def test_from_json_missing_key(self):
        with pytest.raises(KeyError):
            PartitionBoundaries.from_json({"calibration": {"train": [1, 2]}})

    def test_from_json_non_iterable_value(self):
        with pytest.raises((TypeError, KeyError)):
            PartitionBoundaries.from_json({
                "calibration": {"train": 42, "test": [1, 2]},
                "validation": {"train": [1, 2], "test": [3, 4]},
            })

    def test_write_atomic_cleans_up_on_permission_error(self, tmp_path):
        from tools.partitions.fileops import write_atomic
        import os
        target = tmp_path / "readonly_dir" / "file.py"
        target.parent.mkdir()
        target.write_text("original")
        os.chmod(str(target.parent), 0o444)
        try:
            with pytest.raises(OSError):
                write_atomic(target, "new content")
            tmps = list(tmp_path.rglob("*.tmp"))
            assert len(tmps) == 0, f"Orphaned temp files: {tmps}"
        finally:
            os.chmod(str(target.parent), 0o755)


class TestWriteAtomicModePreservation:
    """write_atomic must keep an existing file's permission bits and use a
    umask-respecting default for new files. Regression: a partition bump once
    silently flipped 101 config files 755->644 because os.replace kept the temp
    file's 0o600. See docs/CICs/PartitionFileOps.md.
    """

    def test_write_atomic_preserves_existing_file_mode(self, tmp_path):
        import os
        import stat
        from tools.partitions.fileops import write_atomic

        target = tmp_path / "config_partitions.py"
        target.write_text("old")
        os.chmod(target, 0o755)
        write_atomic(target, "new content")
        assert stat.S_IMODE(target.stat().st_mode) == 0o755
        assert target.read_text() == "new content"

    def test_write_atomic_new_file_uses_umask_default_not_0o600(self, tmp_path):
        import os
        import stat
        from tools.partitions.fileops import write_atomic

        target = tmp_path / "new_lockfile.jsonl"
        write_atomic(target, "x")
        current_umask = os.umask(0)
        os.umask(current_umask)
        expected = 0o666 & ~current_umask
        actual = stat.S_IMODE(target.stat().st_mode)
        assert actual == expected, f"new file mode {oct(actual)} != {oct(expected)}"
        assert actual != 0o600


# ---------------------------------------------------------------------------
# Beige tests: structural compliance
# ---------------------------------------------------------------------------

@pytest.mark.beige
class TestStructuralCompliance:
    """Structural compliance checks for partition tooling."""

    def test_partitions_json_has_required_keys(self):
        import json
        with open(REPO_ROOT / "meta" / "partitions.json") as f:
            data = json.load(f)
        for section in ("calibration", "validation"):
            assert section in data, f"Missing '{section}' in partitions.json"
            for key in ("train", "test"):
                assert key in data[section], f"Missing '{section}.{key}'"
                assert len(data[section][key]) == 2, (
                    f"'{section}.{key}' must be a 2-element list"
                )

    def test_bump_module_has_main_guard(self):
        source = (REPO_ROOT / "tools" / "partitions" / "bump.py").read_text()
        assert 'if __name__ == "__main__"' in source

    def test_domain_has_no_imports_beyond_stdlib(self):
        source = (REPO_ROOT / "tools" / "partitions" / "domain.py").read_text()
        import_lines = [
            ln for ln in source.splitlines()
            if ln.startswith("from ") or ln.startswith("import ")
        ]
        for line in import_lines:
            assert not line.startswith("from tools."), (
                f"domain.py should not import from tools/: {line}"
            )
            assert not line.startswith("from views_"), (
                f"domain.py should not import external packages: {line}"
            )

    def test_fileops_has_no_domain_import(self):
        source = (REPO_ROOT / "tools" / "partitions" / "fileops.py").read_text()
        assert "from tools.partitions.domain" not in source, (
            "fileops.py should not import from domain.py — they are independent"
        )

    def test_all_partition_files_have_generate_function(self):
        files = discover_partition_files(REPO_ROOT)
        for f in files:
            source = f.read_text()
            assert "def generate" in source, (
                f"{f.relative_to(REPO_ROOT)} missing generate() function"
            )


# ---------------------------------------------------------------------------
# Integration tests: bump.py main() end-to-end
# ---------------------------------------------------------------------------

_CANONICAL = {
    "calibration": {"train": [121, 444], "test": [445, 492]},
    "validation": {"train": [121, 492], "test": [493, 540]},
    "steps_default": 36,
}

_PARTITION_SOURCE = '''\
from datetime import date

def _current_month_id():
    today = date.today()
    return (today.year - 1980) * 12 + today.month

def generate(steps=36):
    return {
        "calibration": {"train": (121, 444), "test": (445, 492)},
        "validation": {"train": (121, 492), "test": (493, 540)},
        "forecasting": {
            "train": (121, _current_month_id() - 1),
            "test": (_current_month_id(), _current_month_id() + steps),
        },
    }
'''


@pytest.fixture
def fake_repo(tmp_path):
    """Minimal repo structure for integration tests."""
    repo = tmp_path / "repo"
    meta = repo / "meta"
    meta.mkdir(parents=True)
    (meta / "partitions.json").write_text(_json.dumps(_CANONICAL, indent=2))
    (meta / "fixtures.json").write_text("[]")

    for name in ("alpha", "beta"):
        model = repo / "models" / name
        (model / "configs").mkdir(parents=True)
        (model / "main.py").touch()
        (model / "configs" / "config_partitions.py").write_text(_PARTITION_SOURCE)

    return repo


@pytest.mark.green
class TestBumpIntegration:
    """End-to-end integration tests for bump.py main()."""

    def test_dry_run_modifies_nothing(self, fake_repo, monkeypatch, capsys):
        monkeypatch.setattr(_sys, "argv", ["bump", "--bump", "12"])
        with pytest.raises(SystemExit) as exc:
            bump_main(repo_root=fake_repo)
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "DRY RUN" in out
        assert "Would update 2 files" in out
        for name in ("alpha", "beta"):
            source = (fake_repo / "models" / name / "configs" / "config_partitions.py").read_text()
            assert "(121, 444)" in source

    def test_execute_rewrites_files(self, fake_repo, monkeypatch, capsys):
        monkeypatch.setattr(_sys, "argv", ["bump", "--execute", "--bump", "12"])
        bump_main(repo_root=fake_repo)
        for name in ("alpha", "beta"):
            source = (fake_repo / "models" / name / "configs" / "config_partitions.py").read_text()
            assert "(121, 456)" in source
            assert "(457, 504)" in source

    def test_execute_creates_lockfile(self, fake_repo, monkeypatch, capsys):
        monkeypatch.setattr(_sys, "argv", ["bump", "--execute", "--bump", "12"])
        bump_main(repo_root=fake_repo)
        lockfiles = list((fake_repo / "meta").glob("partition_bump_*.jsonl"))
        assert len(lockfiles) == 1
        entries = [_json.loads(ln) for ln in lockfiles[0].read_text().strip().split("\n")]
        events = [e["event"] for e in entries]
        assert "bump_executed" in events
        assert "bump_completed" in events
        assert entries[0].get("git_commit") is not None

    def test_execute_updates_partitions_json(self, fake_repo, monkeypatch, capsys):
        monkeypatch.setattr(_sys, "argv", ["bump", "--execute", "--bump", "12"])
        bump_main(repo_root=fake_repo)
        updated = _json.loads((fake_repo / "meta" / "partitions.json").read_text())
        assert updated["calibration"]["train"] == [121, 456]
        assert updated["validation"]["test"] == [505, 552]

    def test_preflight_mismatch_blocks(self, fake_repo, monkeypatch, capsys):
        bad = (fake_repo / "models" / "alpha" / "configs" / "config_partitions.py")
        bad.write_text(_PARTITION_SOURCE.replace("(121, 444)", "(121, 400)"))
        monkeypatch.setattr(_sys, "argv", ["bump", "--execute", "--bump", "12"])
        with pytest.raises(SystemExit) as exc:
            bump_main(repo_root=fake_repo)
        assert exc.value.code == 1
        assert "MISMATCH" in capsys.readouterr().out

    def test_missing_partitions_json(self, fake_repo, monkeypatch, capsys):
        (fake_repo / "meta" / "partitions.json").unlink()
        monkeypatch.setattr(_sys, "argv", ["bump", "--bump", "12"])
        with pytest.raises(SystemExit) as exc:
            bump_main(repo_root=fake_repo)
        assert exc.value.code == 1
        assert "not found" in capsys.readouterr().out

    def test_temporal_block(self, fake_repo, monkeypatch, capsys):
        monkeypatch.setattr(_sys, "argv", ["bump", "--bump", "24"])
        with pytest.raises(SystemExit) as exc:
            bump_main(repo_root=fake_repo)
        assert exc.value.code == 1
        assert "exceeds" in capsys.readouterr().out

    def test_override_file_skipped(self, fake_repo, monkeypatch, capsys):
        override_src = "PARTITION_OVERRIDE = True\n\n" + _PARTITION_SOURCE
        (fake_repo / "models" / "alpha" / "configs" / "config_partitions.py").write_text(override_src)
        monkeypatch.setattr(_sys, "argv", ["bump", "--execute", "--bump", "12"])
        bump_main(repo_root=fake_repo)
        alpha_src = (fake_repo / "models" / "alpha" / "configs" / "config_partitions.py").read_text()
        assert "(121, 444)" in alpha_src
        beta_src = (fake_repo / "models" / "beta" / "configs" / "config_partitions.py").read_text()
        assert "(121, 456)" in beta_src

    def test_sync_mode_no_advance(self, fake_repo, monkeypatch, capsys):
        monkeypatch.setattr(_sys, "argv", ["bump", "--execute", "--bump", "0"])
        bump_main(repo_root=fake_repo)
        for name in ("alpha", "beta"):
            source = (fake_repo / "models" / name / "configs" / "config_partitions.py").read_text()
            assert "(121, 444)" in source
        lockfiles = list((fake_repo / "meta").glob("partition_bump_*.jsonl"))
        assert len(lockfiles) == 1
