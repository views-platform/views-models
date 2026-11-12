"""The delivery declaration must describe what already runs (ADR-019, ADR-017 §11 Phase 1).

`deliveries/un_fao.py` is written as a *characterisation*, not a change: nothing reads it
yet, and its job is to say exactly what `postprocessors/un_fao/` does today. The parity
test below is what makes that claim checkable — and it is the reason the later stories
(#347, #348) are safe to attempt at all.

**Parity is scoped to keys that are committed to git** — which, since #127, is all of
them. `COMMITTED_META_KEYS` below is a census asserted for EQUALITY, so every key named
there is under test, `region` included.

⛔ This paragraph said the opposite until 2026-10-30, and the cost was three sessions.
It read: *"Three more (`region`, `wire_contract`, `wire_upload_enabled`) exist only in a
working tree (register C-110) ... the test only pins what the repository can prove."*
That was true before #127 committed them and before ADR-021 made `region` derived, and
the comment above `COMMITTED_META_KEYS` has recorded C-110 as CLOSED since — so this
file has contradicted itself for weeks. On 2026-10-30 three separate sessions read the
stale half, concluded the region was uncommitted, and one of them reported it to the
maintainer as a live risk to the FAO delivery. It is not: `declared_coverage('un_fao')`
and `declared_coverage('un_crafd')` both return `'land_gaul'`, both postprocessor configs
derive `config_meta["region"]` from it with an assert, and both are identical to HEAD.

The lesson is about prose, not code: **a comment describing a defect in the present tense
outlives the defect**, and a file that states both the old and the new state is read as
whichever half the reader meets first.
"""

import ast
import subprocess
from pathlib import Path

import pytest

from tests.conftest import load_config_module

pytestmark = pytest.mark.beige

REPO_ROOT = Path(__file__).resolve().parents[1]
DELIVERIES_DIR = REPO_ROOT / "deliveries"

# Keys the FAO config carries in git.
#
# C-110 is CLOSED as of #127: the config's working-tree-only state was committed, so
# "committed" and "working tree" are now the same set. The comment that used to say
# anything outside this set is working-tree only no longer applies.
COMMITTED_META_KEYS = {
    "name", "algorithm", "targets", "level",
    "ensemble",             # derived from the declaration since #347
    "wire_upload_enabled",  # derived from DELIVERY.intent since #348
    "wire_contract",        # committed by #127
    # Committed by #127, and this guard's own question — "might this key belong in the
    # delivery declaration?" — answered YES. It now IS derived: ADR-021 landed, and the
    # config reads `"region": declared_coverage(CONSUMER)` with an assert two lines below
    # (`postprocessors/un_fao/configs/config_meta.py:97,102`, same shape at `un_crafd:59,69`).
    # `deliveries/status.py:119::declared_coverage` is the single declaration and raises
    # rather than defaulting, so there is no longer a typed literal here to go stale. The
    # un_fao manager reads it at `unfao/managers/unfao.py:236,312,419`.
    "region",
}


FAO_META = "postprocessors/un_fao/configs/config_meta.py"


def _committed_meta_keys() -> set[str]:
    """The FAO meta config's key names, as committed — read *statically*.

    This used to `exec` the file. Since #347 the config imports the delivery
    declaration and manipulates `sys.path`, so executing a detached git blob needs a
    `__file__` that does not exist, and every future import the config gains would
    break this test again. Parsing is enough: the question here is only which keys the
    committed file declares, and `ast` answers that without running anything.
    """
    proc = subprocess.run(
        ["git", "show", f"HEAD:{FAO_META}"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"could not read {FAO_META} from git HEAD.\n"
            f"  git said: {proc.stderr.strip() or '(nothing)'}\n"
            f"  This test inspects the *committed* config, because working-tree-only "
            f"keys differ between checkouts (C-110)."
        )
    try:
        tree = ast.parse(proc.stdout, filename=f"HEAD:{FAO_META}")
    except SyntaxError as exc:
        raise AssertionError(
            f"{FAO_META} does not parse as committed: {exc}.\n"
            f"  Open that file — the committed version is broken, not your working copy."
        ) from exc
    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.add(key.value)
    return keys


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


class TestFaoSourceIsPinned:
    """FAO's source is pinned so a CRAF'd change cannot move it (views-models#599).

    WHY THIS EXISTS. #599 asked for it in as many words: *"Add a test that pins FAO's source to
    `rusty_bucket`, so a CRAF'd change can't move it."* It was the one acceptance item of #599 that
    was never built, found missing on 2026-11-11 while preparing #721.

    The risk is specific and live. Both consumers are declared in the same directory, in files of
    the same shape, and CRAF'd's source has already been repointed once — from `rusty_bucket` to
    `chrome_box` on 2026-10-28, *because* `rusty_bucket` is FAO's and must not go to CRAF'd as a
    stopgap (maintainer's ruling, 2026-10-22). So the edit that moves CRAF'd is one line away from
    the edit that moves FAO, and nothing downstream would notice: `declared_source` reads whatever
    the file says.

    ⛔ These assertions LOAD the module and read the object. They do not match its text. A guard
    that greps for `"rusty_bucket"` in `un_fao.py` is satisfied by the docstring paragraph that
    explains the ruling above — the repository's recurring defect class (register C-189), and the
    reason ADR-025's rule is "execute the code or read a structure, never match its words."
    """

    def test_un_fao_sends_exactly_rusty_bucket(self):
        """One source, named rusty_bucket, at pgm. Not a superset, not a rename."""
        delivery = _load_delivery("un_fao").DELIVERY
        names = [s.name for s in delivery.send]
        assert names == ["rusty_bucket"], (
            f"deliveries/un_fao.py declares send={names!r}, expected exactly ['rusty_bucket'].\n"
            f"  FAO's source is pinned (views-models#599). `rusty_bucket` is FAO's ensemble and "
            f"must not be repointed as a side effect of a CRAF'd change.\n"
            f"  If FAO is genuinely moving source, change it here deliberately and say why in the "
            f"declaration's docstring — that is the review this test exists to force."
        )
        assert [s.level for s in delivery.send] == ["pgm"], (
            f"un_fao's source level is {[s.level for s in delivery.send]!r}, expected ['pgm']. "
            f"FAO is delivered at pgm and curated to land_gaul downstream."
        )

    def test_the_two_consumers_do_not_share_a_source(self):
        """CRAF'd and FAO must not send the same ensemble (maintainer's ruling, 2026-10-22).

        CRAF'd previously sent `rusty_bucket` too; that was repointed to `chrome_box` precisely so
        the two audiences stop sharing one source. This asserts the ruling rather than trusting
        that nobody re-merges them.
        """
        fao = {s.name for s in _load_delivery("un_fao").DELIVERY.send}
        crafd = {s.name for s in _load_delivery("un_crafd").DELIVERY.send}
        shared = fao & crafd
        assert not shared, (
            f"un_fao and un_crafd both send {sorted(shared)!r}. The two consumers must not share a "
            f"source: `rusty_bucket` is FAO's and must not go to CRAF'd as a stopgap (maintainer's "
            f"ruling 2026-10-22, recorded in deliveries/un_crafd.py). "
            f"un_fao sends {sorted(fao)!r}; un_crafd sends {sorted(crafd)!r}."
        )


class TestParityWithCommittedConfig:
    """**Superseded by #347, deliberately not deleted.**

    This class used to assert that `deliveries/un_fao.py` and the FAO config named the
    same source. That was the invariant that made #347 safe to attempt: the declaration
    had to be proven to describe reality before anything depended on it.

    Since #347 the config *derives* its source from the declaration, so comparing the
    two would compare the declaration to itself — a test that always passes and proves
    nothing. Re-adding it would be worse than having no test, because a green
    tautology reads like coverage.

    The invariant that survives is in `tests/test_fao_launcher_reads_declaration.py`:
    changing the declaration must change what the config yields. That is a claim about
    *direction*, which a tautology cannot express.

    What still bites is below: a new key appearing in the committed config.
    """

    def test_parity_scope_is_still_accurate(self):
        """If the FAO config gains or loses a committed key, this must be revisited.

        A new committed key might belong in the delivery declaration, and silently
        ignoring it is how the two drift apart again.
        """
        committed = _committed_meta_keys()
        assert committed == COMMITTED_META_KEYS, (
            f"the committed keys of {FAO_META} changed: "
            f"{sorted(committed ^ COMMITTED_META_KEYS)}.\n"
            f"  Decide whether the new/removed key belongs in deliveries/un_fao.py, "
            f"then update COMMITTED_META_KEYS in this file."
        )


# ── The additive guard has been retired ───────────────────────────────────
#
# #343 asserted that nothing under postprocessors/, models/, ensembles/ or
# monthly_run.sh referenced deliveries/, because that story was additive. #347 is the
# story that changed it, and the guard fired there exactly as intended.
#
# It was not deleted, it was **inverted**: the same fact is still checked, with the
# opposite expected value, in
# tests/test_fao_launcher_reads_declaration.py::TestTheAdditiveGuardIsCorrectlyRetired.
# A guard that is removed when it becomes inconvenient teaches the next person that
# guards are negotiable.
