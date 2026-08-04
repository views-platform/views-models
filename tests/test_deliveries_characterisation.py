"""The delivery declaration must describe what already runs (ADR-019, ADR-017 §11 Phase 1).

`deliveries/un_fao.py` is written as a *characterisation*, not a change: nothing reads it
yet, and its job is to say exactly what `postprocessors/un_fao/` does today. The parity
test below is what makes that claim checkable — and it is the reason the later stories
(#347, #348) are safe to attempt at all.

**Parity is scoped to keys that are committed to git.** `git show HEAD:` of the FAO
config declares five keys — name, algorithm, targets, level, ensemble. Three more
(`region`, `wire_contract`, `wire_upload_enabled`) exist only in a working tree
(register C-110), and a test that asserted against them would pass on one checkout and
fail on another. The declaration still records `coverage`, because that is what runs;
the *test* only pins what the repository can prove.
"""

import subprocess
from pathlib import Path

import pytest

from tests.conftest import load_config_module

pytestmark = pytest.mark.beige

REPO_ROOT = Path(__file__).resolve().parents[1]
DELIVERIES_DIR = REPO_ROOT / "deliveries"

# Keys the FAO config carries in git. Anything outside this set is working-tree only
# (C-110) and must not be asserted against — see the module docstring.
COMMITTED_META_KEYS = {"name", "algorithm", "targets", "level", "ensemble"}


FAO_META = "postprocessors/un_fao/configs/config_meta.py"


def _committed_fao_meta() -> dict:
    """The FAO meta config as committed, not as it sits in someone's working tree."""
    proc = subprocess.run(
        ["git", "show", f"HEAD:{FAO_META}"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"could not read {FAO_META} from git HEAD.\n"
            f"  git said: {proc.stderr.strip() or '(nothing)'}\n"
            f"  This test compares the delivery declaration against the *committed*\n"
            f"  config, because working-tree-only keys differ between checkouts (C-110)."
        )
    namespace: dict = {}
    try:
        exec(compile(proc.stdout, f"<HEAD:{FAO_META}>", "exec"), namespace)  # noqa: S102
    except SyntaxError as exc:
        raise AssertionError(
            f"{FAO_META} does not parse as committed: {exc}.\n"
            f"  Open that file — the committed version is broken, not your working copy."
        ) from exc
    return namespace["get_meta_config"]()


def _load_delivery(consumer: str):
    path = DELIVERIES_DIR / f"{consumer}.py"
    assert path.exists(), (
        f"no delivery declaration for '{consumer}'.\n"
        f"  Expected: {path.relative_to(REPO_ROOT)}\n"
        f"  See docs/ADRs/019_delivery_declaration.md §1 for the file's shape."
    )
    return load_config_module(path, module_name=f"delivery_{consumer}")


# ── The declaration exists and is well formed ───────────────────────────────


class TestDeclarationShape:
    def test_deliveries_package_exists(self):
        assert DELIVERIES_DIR.is_dir(), (
            "deliveries/ does not exist. ADR-017 §3 puts the delivery edge in "
            "deliveries/<consumer>.py, never on the source."
        )

    def test_un_fao_declares_delivery_and_require(self):
        mod = _load_delivery("un_fao")
        assert hasattr(mod, "DELIVERY"), "deliveries/un_fao.py must define DELIVERY"
        assert hasattr(mod, "REQUIRE"), "deliveries/un_fao.py must define REQUIRE"

    def test_filename_is_the_consumer(self):
        """ADR-019 §1: no `to` key may repeat the filename, so the two cannot disagree."""
        mod = _load_delivery("un_fao")
        assert not hasattr(mod.DELIVERY, "to"), (
            "DELIVERY must not carry a `to` key — the filename is the consumer "
            "(ADR-019 §8, first rejected alternative)."
        )


# ── Parity: the declaration equals what is committed ────────────────────────


class TestParityWithCommittedConfig:
    """The declaration must describe reality before anything is allowed to depend on it."""

    def test_ensemble_matches(self):
        mod = _load_delivery("un_fao")
        committed = _committed_fao_meta()
        declared = {s.name for s in mod.DELIVERY.send}
        assert declared == {committed["ensemble"]}, (
            f"deliveries/un_fao.py sends {sorted(declared)} but "
            f"postprocessors/un_fao/configs/config_meta.py declares "
            f"'{committed['ensemble']}'.\n"
            f"  These must agree until #347 makes the launcher read the declaration."
        )

    def test_level_matches(self):
        mod = _load_delivery("un_fao")
        committed = _committed_fao_meta()
        for source in mod.DELIVERY.send:
            assert source.level == committed["level"], (
                f"deliveries/un_fao.py claims {source.level}('{source.name}') but "
                f"postprocessors/un_fao/configs/config_meta.py declares "
                f"level '{committed['level']}'."
            )

    def test_targets_match(self):
        mod = _load_delivery("un_fao")
        committed = _committed_fao_meta()
        assert list(mod.REQUIRE.targets) == list(committed["targets"]), (
            f"deliveries/un_fao.py requires {list(mod.REQUIRE.targets)} but "
            f"postprocessors/un_fao/configs/config_meta.py declares "
            f"{committed['targets']}."
        )

    def test_parity_scope_is_still_accurate(self):
        """If the FAO config gains or loses a committed key, this test must be revisited.

        Guards the assumption in the module docstring: a new committed key might belong
        in the declaration, and silently ignoring it is how the two drift apart.
        """
        committed = set(_committed_fao_meta())
        assert committed == COMMITTED_META_KEYS, (
            f"the committed keys of postprocessors/un_fao/configs/config_meta.py "
            f"changed: {sorted(committed ^ COMMITTED_META_KEYS)}.\n"
            f"  Decide whether the new/removed key belongs in deliveries/un_fao.py, "
            f"then update COMMITTED_META_KEYS in this file."
        )


# ── Nothing depends on the declaration yet ─────────────────────────────────


class TestNoBehaviourChange:
    def test_nothing_reads_deliveries_yet(self):
        """#343 is additive. The launcher starts reading the declaration in #347."""
        proc = subprocess.run(
            ["git", "grep", "-l", "-E", r"deliveries[./]", "--",
             "postprocessors/", "models/", "ensembles/", "monthly_run.sh"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        # git grep: 0 = matches found, 1 = none, anything else = it did not run.
        # Reading empty stdout as "no matches" would make this guard pass on error —
        # the defect recorded as C-113, in the one test that proves this story is
        # additive. Silence must not be mistaken for a clean result.
        assert proc.returncode in (0, 1), (
            f"git grep could not run (rc={proc.returncode}), so this guard proved "
            f"nothing.\n  git said: {proc.stderr.strip() or '(nothing)'}"
        )
        hits = proc.stdout.split()
        assert not hits, (
            f"{hits} reference deliveries/ — but #343 must change no behaviour.\n"
            f"  Making the launcher read the declaration is #347."
        )
