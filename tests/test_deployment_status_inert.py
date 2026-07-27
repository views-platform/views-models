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

# Explicit, reasoned exceptions. Empty by design — a non-empty entry is a
# deliberate decision that must be reconciled with ADR-017.
ALLOWLIST: set[tuple[str, int]] = set()


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
        for i, line in enumerate(text.splitlines(), start=1):
            if _COMPARISON.search(line) and (str(py), i) not in ALLOWLIST:
                hits.append(f"{py}:{i}: {line.strip()}")
    return hits


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
