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
from packaging.specifiers import SpecifierSet

pytestmark = pytest.mark.green

REPO_ROOT = Path(__file__).resolve().parents[1]


# ── the one deferred decision, named, with its reason ─────────────────
# views-r2darts2 is declared three mutually different ways across 31 models:
#   ==0.1.0          (12)  the CM datafactory label models, verified end-to-end
#                          at that exact version in PR #232
#   >=0.1.0          (10)  unbounded
#   >=1.0.0,<2.0.0    (9)
# Collapsing these is NOT hygiene: they are three true statements about an
# upstream whose versioning is unsettled (the cached_path fix is still
# uncommitted, r2darts#22). Forcing one spec would simplify the files by lying
# about the world. Registered as part of C-115; revisit when r2darts 1.x is
# published AND r2darts#22 is committed — that is the named trigger, not "later".
DEFERRED_PACKAGES = {
    "views-r2darts2": (
        "FOUR specs across 52 declarations, measured 2026-10-06: >=0.1.0 x21, >=0.2.0 x14, "
        ">=1.0.0,<2.0.0 x9, [manager]>=0.2.3,<0.3.0 x8. The last of those is the only one "
        "that resolves views-pipeline-core on 0.2.x, where it is optional — #531 gave it to "
        "eight models; see test_the_manager_extra_is_present_wherever_0_2_x_is_reachable for "
        "the 35 that still lack it. The >=1.0.0,<2.0.0 nine match NO published version "
        "(upstream is at 0.2.4), so they resolve to nothing at all — a separate defect. "
        "The previous reason here named the TRIGGER 'views-r2darts2 1.x published AND "
        "r2darts#22 committed'. Both halves were unreachable: upstream went 0.2.x and never "
        "1.x, and r2darts#22's data-path half turned out to be already fixed in 0.2.0 "
        "(`_resolve_raw_parquet_path`, verified against the 0.2.3 tag), so it will not be "
        "'committed' as described. A trigger that cannot fire is a permanent exemption. "
        "REPLACEMENT TRIGGER: this branch adopting one spec for all r2darts2 tenants, as "
        "`development` did in #485 — at which point this entry is DELETED, not amended, "
        "because the divergence it describes will no longer exist. See C-115, C-116."
    ),
}

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


# ── the `manager` extra: why its absence is silent, and who is still missing it ──────
#
# views-r2darts2 moved `views-pipeline-core` from a HARD dependency to an OPTIONAL one at
# 0.2.0, supplied only by the `manager` extra. Measured from the published metadata:
#
#     0.1.1   views-pipeline-core>=2.0.0,<3.0.0     (hard — no extras at all)
#     0.2.0+  views-pipeline-core>=3.0.0,<4.0.0 ; extra == "manager"
#
# Every model's `main.py` opens with `from views_pipeline_core...`. So a model whose spec can
# resolve to 0.2.0 or later and does NOT declare `[manager]` installs no pipeline-core and
# dies on its first import — `ModuleNotFoundError: No module named 'views_pipeline_core'`.
# That is views-models **#531**, reported by Dylan on 2026-10-01 against `crimson_tide`.
#
# It is silent twice over. `pip install -r requirements.txt` succeeds, and a model sharing a
# conda prefix with a co-tenant that DID install pipeline-core inherits it (C-116) — so the
# same declaration works or fails depending on which model ran in that prefix first.
#
# A spec of `>=0.1.0` is enough to trigger it: unbounded, so pip takes 0.2.3.
#: Every published 0.2.x, plus the tagged-but-unpublished 0.2.4. A specifier that admits
#: any of these can resolve to a version where views-pipeline-core is optional.
_ZERO_TWO_RELEASES = ("0.2.0", "0.2.1", "0.2.2", "0.2.3", "0.2.4")

KNOWN_MISSING_MANAGER_EXTRA = {
    "bad_romance",
    "blue_ocean",
    "brave_heart",
    "bright_star",
    "cold_heart",
    "dancing_monkey",
    "dancing_queen",
    "dark_necessities",
    "dark_river",
    "elastic_heart",
    "free_fallin",
    "golden_eagle",
    "good_life",
    "heat_waves",
    "little_talks",
    "mister_bluesky",
    "new_rules",
    "old_rules",
    "rapid_fire",
    "ravaging_cleric",
    "ravaging_fighter",
    "ravaging_mage",
    "ravaging_thief",
    "red_hawk",
    "revolving_door",
    "roaming_cleric",
    "roaming_fighter",
    "roaming_mage",
    "roaming_thief",
    "silent_fox",
    "smol_cat",
    "warring_cleric",
    "warring_fighter",
    "warring_mage",
    "warring_thief",
}


def test_the_manager_extra_is_present_wherever_0_2_x_is_reachable():
    """Characterization test. It asserts nothing about what the set SHOULD be — only that
    changing it is deliberate (the same contract as
    `test_environment_sharing_is_recorded_not_discovered`).

    Two directions, both wanted:

    - **A model losing its `[manager]`** joins the set and turns this red. That is the
      regression guard for #531: the eight models fixed there must keep the extra, and
      nothing else in this suite would notice if one lost it.
    - **A model gaining it** leaves the set and also turns this red, asking for the set to
      shrink. Progress has to be recorded, not absorbed.

    The set is large because the defect is upstream, not per-model:
    `views_pipeline_core/templates/model/template_requirement_txt.py` writes a bare
    `{package}=={version}` with no extras, so **every r2darts2 model this scaffold has ever
    generated was born with it**. That is why #485 had to retrofit 31 by hand on
    `development`. Fixing the remaining models here is deliberately NOT part of #531 — that
    PR moved eight models out of the HydraNet prefix and is scoped to them.

    Trigger for emptying this set: the scaffold template emits extras, or a dedicated PR
    retrofits the rest on this branch.
    """
    actual = set()
    for name, _number, req in _declarations():
        if req.name != "views-r2darts2":
            continue
        # Every 0.2.x that exists, not just the endpoints. Testing only "0.2.0" and
        # "0.2.3" let `==0.2.1`, `==0.2.2`, `>=0.2.1,<0.2.3` and `==0.2.4` through — each
        # of which needs the extra just as much. Found reviewing this test, not by it.
        reaches_02 = any(
            SpecifierSet(str(req.specifier)).contains(v) for v in _ZERO_TWO_RELEASES
        )
        if reaches_02 and "manager" not in req.extras:
            actual.add(Path(name).parent.name if "/" in name else name)

    assert actual == KNOWN_MISSING_MANAGER_EXTRA, (
        "the set of models missing the `[manager]` extra changed.\n"
        f"  newly missing (REGRESSION — these cannot import views_pipeline_core): "
        f"{sorted(actual - KNOWN_MISSING_MANAGER_EXTRA)}\n"
        f"  newly fixed (good — remove them from KNOWN_MISSING_MANAGER_EXTRA): "
        f"{sorted(KNOWN_MISSING_MANAGER_EXTRA - actual)}\n"
        "See views-models#531 and the comment above this test."
    )
