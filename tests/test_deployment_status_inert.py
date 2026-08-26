"""Pin the ADR-017 invariant: `deployment_status` never branches deployed-vs-shadow.

A falsification audit (2026-07-26) established that, on a standalone model, the
choice between `deployed` and `shadow` changes no behaviour anywhere in the
platform: the field is read only by the config sniffer (validate membership +
block `deprecated`), the provenance log-stamp (a write), the ensemble guard
(ensemble path; branches only on `deprecated`/dead `production`), and the
catalog/README display. No code compares `deployment_status` to the literals
`"deployed"` or `"shadow"`.

This test guards that invariant so a future change cannot silently give the
label teeth — which would invalidate ADR-017's "derived, never declared" model.
If it fails, either the new behavioural branch is a mistake, or ADR-017 and this
test must be updated together (and the branch added to ALLOWLIST with a reason).

**Amended 2026-08-04 (#344).** That second path was taken, once. ADR-017 is now
being implemented, and §3's migration mapping must read `deployed` in order to
translate it into the maturity vocabulary. `deliveries/coherence.py` is therefore
the single declared reader, and the invariant is now "exactly one migration point
may read these values, nothing else may" — see ALLOWLIST. ADR-017 §2 records the
same change. Two further tests keep the exemption honest: it must name a file that
exists, and that file must still need it.

Green-team (ADR-005): pure source scan, no ML deps. Scans this repo always, and
the installed `views_pipeline_core` package when importable (skip-truthful, C-75).
"""
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.green

REPO_ROOT = Path(__file__).resolve().parent.parent

# A behavioural branch requires comparing the field to the literal value.
# The allowed-set definition (`{"shadow", "deployed", ...}`) and the log-stamp
# default (`get("deployment_status", "shadow")`) use neither `==` nor `!=`, so
# they do not match. `deprecated`/`production` are out of scope (they are not the
# deployed-vs-shadow distinction this invariant is about).
_COMPARISON = re.compile(
    r"""(?:==|!=)\s*['"](?:deployed|shadow)['"]"""      # x == "shadow"
    r"""|['"](?:deployed|shadow)['"]\s*(?:==|!=)"""     # "deployed" != y
)

# Explicit, reasoned exceptions, keyed by path relative to the repo root.
#
# Keyed by FILE, not (file, line): a line number rots the moment anything above it
# is edited, and a stale exemption silently re-opens the hole it was guarding.
#
# The invariant this test pins has changed shape, and the change is deliberate.
# Before 2026-08-04 the claim was "nothing anywhere branches deployed-vs-shadow",
# which was ADR-017 §2's evidence that the label is inert. ADR-017 is now being
# implemented (#342), and ADR-017 §3 defines a migration mapping that must read the
# old value in order to translate it. So the claim becomes:
#
#     Exactly one declared migration point may read these values. Nothing else may.
#
# That is a stronger invariant than an empty allowlist would be today, because it
# still fails for every other file — including any attempt to make training,
# forecasting or delivery depend on the label without going through the ADR.
ALLOWLIST: dict[str, str] = {
    "deliveries/coherence.py": (
        "ADR-017 §3's migration mapping: `deployed` -> `graduate` only where R2 "
        "already holds, else `candidate`. Translating the old vocabulary requires "
        "reading it. Retire this entry when Phase 2 lands the rename "
        "(views-pipeline-core#398) and the old values no longer exist."
    ),
}


def _offending_lines(root: Path) -> list[str]:
    hits: list[str] = []
    for py in root.rglob("*.py"):
        parts = set(py.parts)
        if {".git", "__pycache__", "tests", "test"} & parts:
            continue
        if py.name.startswith("test_") or py.name == __file__.rsplit("/", 1)[-1]:
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        try:
            relative = py.relative_to(root).as_posix()
        except ValueError:
            relative = py.as_posix()
        if relative in ALLOWLIST:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if _COMPARISON.search(line):
                hits.append(f"{py}:{i}: {line.strip()}")
    return hits


def test_every_allowlisted_file_still_exists():
    """A stale exemption silently re-opens the hole it was guarding.

    If an allowlisted file is deleted or renamed, the entry must go with it —
    otherwise a future file at that path inherits an exemption nobody granted it.
    """
    missing = [rel for rel in ALLOWLIST if not (REPO_ROOT / rel).exists()]
    assert not missing, (
        f"ALLOWLIST names files that no longer exist: {missing}.\n"
        f"  Open tests/test_deployment_status_inert.py and remove the entries."
    )


def test_allowlisted_file_actually_needs_its_exemption():
    """An exemption for a file that no longer branches is dead weight — and it
    would silently cover a *new* branch added to that file later."""
    unnecessary = []
    for rel in ALLOWLIST:
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        if not any(_COMPARISON.search(line) for line in path.read_text().splitlines()):
            unnecessary.append(rel)
    assert not unnecessary, (
        f"these files are allowlisted but no longer branch on deployed/shadow: "
        f"{unnecessary}.\n"
        f"  Remove them from ALLOWLIST in tests/test_deployment_status_inert.py — "
        f"an exemption wider than its reason is how the invariant erodes."
    )


def test_views_models_never_branches_on_deployed_or_shadow():
    """No file in this repo compares `deployment_status` to 'deployed'/'shadow'."""
    hits = _offending_lines(REPO_ROOT)
    assert not hits, (
        "ADR-017 invariant broken — code now branches deployed-vs-shadow "
        "(the label was inert; something gave it teeth):\n" + "\n".join(hits)
    )


def test_pipeline_core_never_branches_on_deployed_or_shadow():
    """Same invariant in the installed views_pipeline_core (the real risk site)."""
    try:
        import views_pipeline_core
    except ImportError:
        pytest.skip("views_pipeline_core not installed — cannot scan (C-75 truthful skip)")
    pkg_root = Path(views_pipeline_core.__file__).resolve().parent
    hits = _offending_lines(pkg_root)
    assert not hits, (
        "ADR-017 invariant broken in views_pipeline_core — a deployed-vs-shadow "
        "branch appeared:\n" + "\n".join(hits)
    )
