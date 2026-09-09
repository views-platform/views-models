"""Every `requirements.txt` in this repo is parseable, bounded, and consistent.

131 of these files are maintained by hand, one or two lines each, and nothing has
ever checked them. What that cost, measured 2026-08-02:

    models/fake_model      `views-stepshifter==>=1.0.0,<2.0.0` — unparseable, and
                           the file therefore could not install at all (#316)
    27 files               `views-datafactory>=1.9.0` with no ceiling, so a 2.0
                           release would install itself during a monthly run
    3 files                no trailing newline — which caused a wrong conclusion
                           during the very session that added this test, when a
                           `cat` of all 131 glued adjacent files together and the
                           result was read as corrupted requirement lines

**Why an allowlist appears here at all, and why it has one entry.** These rules were
written in the order above deliberately: parse (failed on 1 file, fixed), newline
(failed on 3, fixed), ceiling (failed on 37, of which 27 fixed). Everything that
could be fixed was fixed *before* this test landed, so the exception list is not a
way to make a red test green — it is the residue that a decision was deliberately
deferred on. Today that residue is one package. If it ever exceeds two, the honest
reading is that this test has become somewhere to hide, and it should be deleted
rather than extended (register **D-06**).

Coverage this test does NOT claim: it reads declarations, never environments. A
declaration and the environment it names disagree in both directions in this repo
(**C-116**) and no test over these files can see that.
"""

from pathlib import Path
import subprocess

import pytest

from packaging.requirements import InvalidRequirement, Requirement

pytestmark = pytest.mark.green

REPO_ROOT = Path(__file__).resolve().parents[1]


# ── no deferred packages ──────────────────────────────────────────────
# This held one entry — views-r2darts2, declared three mutually different ways
# across 31 models — deferred on the grounds that the three specs were "three true
# statements about an upstream whose versioning is unsettled".
#
# Measured 2026-09-09 (#317), and the premise did not survive:
#   - `==0.1.0` was NEVER on PyPI. The tag exists; the publish workflow fires on
#     GitHub Release and 0.1.0 never got one. The pin was written from `pip list`
#     against a local editable install, three months before the tag existed.
#   - `>=1.0.0,<2.0.0` has never been satisfiable either. No 1.x tag, branch,
#     milestone or issue exists.
#   - All 31 models import the same `DartsForecastingModelManager` with the same
#     constructor. The three specs encoded drift, not three statements about
#     anything. Two of the three named a version that does not exist.
#
# The named trigger — "views-r2darts2 1.x published AND r2darts#22 committed" —
# could not fire: upstream went 0.2.x, so the first condition was unreachable by
# construction. A deferral whose trigger cannot fire is a permanent exemption.
#
# All 31 now declare `views-r2darts2>=0.1.1,<0.2.0` — one spec, and the first in this
# repo's history that resolves for all of them. 0.1.1 is the only released version in
# that range and it installs cleanly on a bare machine (verified 2026-09-10 in an empty
# venv: pip exit 0, and all three of main.py's imports execute).
#
# 0.2.x is NOT adoptable, and the reason is not ours to fix. `views-r2darts2[manager]`
# cannot resolve against ANY published views-pipeline-core — four independent conflicts,
# of which wandb is only the first pip reports. The blocking one is pandas: r2darts2
# 0.2.x requires darts==0.46.1 (pandas>=2.2) while pipeline-core's own dependencies cap
# pandas<2.0. That is the platform-wide pandas lock (pipeline-core #308 / C-112).
# Reported upstream as views-r2darts2#34, #35, #36.
#
# The cost of 0.1.1, stated plainly: it pins views-pipeline-core<3.0.0, so these 31 are
# the only models in the fleet still on pipeline-core 2.x. That split stands until the
# pandas migration lands. It is deliberate, and it beats 31 models that cannot install.
DEFERRED_PACKAGES: dict[str, str] = {}

# pip accepts a bare VCS URL as a requirements.txt line; PEP 508 does not, because
# such a line names no package. `apis/un_fao/requirements.txt` uses that form. It is
# valid pip input, so it is carved out rather than "fixed" — but it is invisible to
# every rule below, which is the actual argument for the `name @ url` form instead.
_BARE_URL_PREFIXES = ("git+", "http://", "https://", "-e ", "-r ", "--")


def _requirements_files():
    out = subprocess.run(
        ["git", "ls-files", "-z", "*requirements.txt"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    for name in out.split("\0"):
        if name:
            path = REPO_ROOT / name
            if path.is_file():
                yield name, path


def _declarations():
    """(file, line number, Requirement) for every parseable, non-URL line."""
    for name, path in _requirements_files():
        for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith(_BARE_URL_PREFIXES):
                continue
            try:
                yield name, number, Requirement(line)
            except InvalidRequirement:
                continue  # reported by the parse test, not swallowed


def test_every_requirement_line_parses():
    """An unparseable line means the file cannot install — the loudest failure, unnoticed."""
    bad = []
    for name, path in _requirements_files():
        for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith(_BARE_URL_PREFIXES):
                continue
            try:
                Requirement(line)
            except InvalidRequirement as exc:
                bad.append(f"  {name}:{number}: {line!r} — {str(exc).splitlines()[0]}")
    assert not bad, "unparseable requirement lines:\n" + "\n".join(bad)


def test_every_file_ends_with_a_newline():
    """A file without one silently concatenates with the next when tools read in bulk."""
    missing = [
        name for name, path in _requirements_files()
        if path.stat().st_size and path.read_bytes()[-1:] != b"\n"
    ]
    assert not missing, (
        "no trailing newline — reading these in bulk glues them to the next file:\n"
        + "\n".join(f"  {n}" for n in missing)
    )


def test_no_dependency_is_declared_without_an_upper_bound():
    """An unbounded spec silently accepts the next breaking major of an internal package."""
    unbounded = []
    for name, number, req in _declarations():
        if req.name in DEFERRED_PACKAGES or req.url:
            continue
        if not any(s.operator in ("<", "<=", "==", "===", "~=") for s in req.specifier):
            unbounded.append(f"  {name}:{number}: {req}")
    assert not unbounded, (
        "no upper bound — a major release installs itself on the next monthly run:\n"
        + "\n".join(unbounded)
        + "\nAdd a ceiling, or record the package in DEFERRED_PACKAGES with a reason."
    )


def test_a_package_is_declared_the_same_way_everywhere():
    """Divergent specs for one package are decided by run order, not by intent.

    131 files resolve into 11 shared environments (**C-116**), so two tenants
    declaring the same package differently do not each get what they asked for —
    whichever ran last wins, and pip reports success to both.
    """
    specs = {}
    for name, number, req in _declarations():
        if req.name in DEFERRED_PACKAGES:
            continue
        # A URL requirement (`name @ git+...`) carries no version specifier at all, so
        # comparing it against a versioned declaration always "diverges" -- a category
        # error, not a finding. postprocessors/un_fao pins views-datafactory to a git
        # branch this way. That IS worth attention (a branch pointer moves under you),
        # but it is a different concern from two versions of one package in one shared
        # environment, which is what this rule exists to catch.
        if req.url:
            continue
        specs.setdefault(req.name, {}).setdefault(str(req.specifier), []).append(name)

    divergent = {pkg: v for pkg, v in specs.items() if len(v) > 1}
    assert not divergent, (
        "one package declared several ways:\n"
        + "\n".join(
            f"  {pkg}:\n"
            + "\n".join(f"      {spec or '(none)'}  x{len(files)}  e.g. {files[0]}"
                        for spec, files in sorted(variants.items()))
            for pkg, variants in sorted(divergent.items())
        )
        + "\nUnify them, or record the package in DEFERRED_PACKAGES with a reason."
    )


def test_the_deferred_list_stays_small_enough_to_be_honest():
    """The allowlist's length is the metric — past two it is a hiding place (D-06)."""
    assert len(DEFERRED_PACKAGES) <= 2, (
        f"{len(DEFERRED_PACKAGES)} packages are exempted from the rules above. "
        "At this size the exemptions are the policy. Fix them, or delete this "
        "test rather than keep extending it — see register D-06."
    )
    for package, reason in DEFERRED_PACKAGES.items():
        assert "trigger" in reason.lower(), (
            f"{package} is deferred without a named trigger for revisiting it. "
            "CLAUDE.md: defer behind a named trigger, never a vague 'later'."
        )
