"""The reconciliation suites must SKIP without pipeline-core 3.0.0, never fail to collect.

`reconciliation/` is written against `views_pipeline_core.domain.reconciliation_port`,
which exists only from pipeline-core **3.0.0** — unreleased. CI installs the *published*
core (2.3.1), so the symbol is absent there.

**What went wrong.** `reconciliation/__init__.py:11` imports `composition`, which imports
that port at module level. So importing *any* reconciliation submodule fails, including
`country_mapping` and `source_detection`, which have no pipeline-core dependency of their
own. Three of the seven test modules already called `pytest.importorskip`, but the call
sat *below* a module-level `from reconciliation... import`, so it never ran. The result was
`Interrupted: 7 errors during collection` — which aborts the **entire** run, not just those
files. `run_tests.yml` was therefore red on every commit, and had been for the whole period
in which this repo was being actively worked on.

**The invariant this test enforces:** in every `test_reconciliation_*.py`, the guard comes
*before* the first reconciliation import. That single ordering is the whole difference
between a truthful skip and a red pipeline.

**Verified dynamically when this landed**, by simulating CI with a meta-path finder that
hides the port: 7 skipped with the symbol absent, 32 passed with it present. That check is
not repeated here — running pytest inside pytest to re-prove it every time would cost more
than it is worth, and the ordering assertion below is what actually broke.

**This test becomes obsolete when pipeline-core 3.0.0 ships.** At that point the guards stop
skipping anything and can be removed; the tests will simply run. Nothing here needs to
change first — a guard that never triggers is inert, not wrong.
"""

from pathlib import Path
import re

import pytest

pytestmark = pytest.mark.green

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"

PORT = "views_pipeline_core.domain.reconciliation_port"
_GUARD = re.compile(r'^pytest\.importorskip\(\s*["\']' + re.escape(PORT) + r'["\']', re.M)
_IMPORT = re.compile(r"^(?:from reconciliation[\w.]*\s+import|import reconciliation)", re.M)


def _reconciliation_suites():
    """The suites under guard — excluding this file, which matches its own glob.

    Same self-exclusion as `tests/test_seam_contract_citations.py`: a file that
    describes a rule necessarily contains the strings the rule is about.
    """
    me = Path(__file__).name
    found = sorted(p for p in TESTS_DIR.glob("test_reconciliation_*.py") if p.name != me)
    assert found, "no test_reconciliation_*.py found — did they move?"
    return found


@pytest.mark.parametrize(
    "path", _reconciliation_suites(), ids=lambda p: p.name
)
def test_guard_precedes_the_first_reconciliation_import(path):
    """Below the import, `importorskip` cannot run — collection errors instead."""
    source = path.read_text(encoding="utf-8")

    guard = _GUARD.search(source)
    assert guard, (
        f"{path.name} imports reconciliation without guarding on {PORT}.\n"
        f"Add, above the first reconciliation import:\n"
        f'    pytest.importorskip("{PORT}")'
    )

    first_import = _IMPORT.search(source)
    assert first_import, f"{path.name} matches test_reconciliation_* but imports no reconciliation module"

    guard_line = source[: guard.start()].count("\n") + 1
    import_line = source[: first_import.start()].count("\n") + 1
    assert guard_line < import_line, (
        f"{path.name}: the guard is on line {guard_line} but the first reconciliation "
        f"import is on line {import_line}. Below the import the guard never executes, and "
        f"collection ERRORS instead of skipping — which aborts the whole pytest run, not "
        f"just this file."
    )


def test_the_package_init_is_why_every_submodule_is_affected():
    """Records the mechanism, so 'but this module has no pipeline-core dependency' is answered.

    If this ever fails because `__init__.py` stopped importing `composition`, the guards may
    be narrowable — check before deleting any of them.
    """
    init = (REPO_ROOT / "reconciliation" / "__init__.py").read_text(encoding="utf-8")
    assert "from reconciliation.composition import" in init, (
        "reconciliation/__init__.py no longer imports composition. The seven guards were "
        "added because it did, which made every submodule import pull the unreleased port. "
        "Re-check whether they are all still needed."
    )
