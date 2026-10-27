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

import re
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
# 2026-09-10 (#317): all 31 declared `views-r2darts2>=0.1.1,<0.2.0` — one spec, the first in
# this repo's history that resolved for all of them. 0.2.x was NOT adoptable then:
# `views-r2darts2[manager]` could not resolve against any published views-pipeline-core —
# wandb (pipeline-core capped <0.19, r2darts2 floored >=0.28.2) and pandas (r2darts2 0.2.x
# required darts 0.46 = pandas>=2.2; viewser holds pandas<2). Reported as views-r2darts2#34,
# #35, #36; the cost was that these 31 sat on pipeline-core 2.x while the fleet moved to 3.x.
#
# 2026-09-19 (#485): both walls fell on the same morning, from both sides. pipeline-core 3.3.0
# widened wandb to <1.0 (their ADR-067); views-r2darts2 0.2.3 pins darts==0.40.0 and
# pandas<2. All 31 now declare `views-r2darts2[manager]>=0.2.3,<0.3.0` — the `[manager]`
# extra is what brings views-pipeline-core (>=3.0,<4) on 0.2.x, where it is optional — and
# it resolves from PyPI beside pipeline-core 3.3.0 and viewser 6.6.4 (measured in a clean
# venv; one model run end-to-end on the published wheel). The 2.x split is over.
# views-datafactory is split DELIBERATELY and temporarily, 2026-09-30 (#509).
#
# The credential-handling fixes landed in 1.13.0: before it the client could carry a netrc
# credential across a redirect to another host, and could embed it in error messages. Every
# leg that runs on RENTED HARDWARE holds that credential, so the three that do are floored at
# 1.13.0 — both postprocessor requirement files and the pod install in
# tools/podrun/pod_run_model.sh (not a requirements file, so invisible to this rule).
#
# The other 36 declarations are model requirements at >=1.9.0. They are NOT floored here, for
# two reasons. They are unrelated to the delivery legs, so raising them belongs to #509 rather
# than to a delivery PR; and one of them is `models/violet_visitor/requirements.txt`, whose
# contents another session owns and which this session is instructed not to edit.
#
# The divergence is therefore real and is not decided by run order in the way C-116 warns
# about: the two postprocessors share one prefix and agree with each other, and the models
# resolve elsewhere. The resolver picks 1.13.0 for all of them today regardless — the floor
# only forbids something lower.
#
# COST OF THIS ENTRY, because it is wider than it looks: a DEFERRED_PACKAGES entry also
# exempts the package from test_no_dependency_is_declared_without_an_upper_bound, so nothing
# here would notice `<2.0.0` being dropped from a views-datafactory line while this entry
# stands. That hole is closed explicitly by
# tests/test_falsification_40_lesson_run_readiness.py::test_every_datafactory_declaration_
# keeps_its_upper_bound, which is narrower than this rule and survives the deferral.
#
# REMOVE THIS ENTRY when #509 floors the remaining 36 — that is the whole of the trigger.
#
# The count in that sentence is a TRIGGER, not decoration: acting on a stale one deletes a
# live exemption. It said 34 and was right until amber_kestrel's requirements.txt made it
# 36 declarations, 35 of them unfloored — the kind of drift that is only ever found by
# someone reading carefully. It is now asserted against the measured count below, so it
# cannot go stale silently again, and copper_kestrel's requirements.txt took it to 37 and 36
# on 2026-10-16 — the assertion is what reported that, not a reader. The three seed
# replicates of amber_kestrel took it to 40 and 39 on 2026-10-19 (views-models#567); their
# requirements.txt files are byte-identical copies of amber_kestrel's, which is asserted by
# tests/test_roster_conformance.py and is load-bearing rather than tidy — a dependency
# difference between replicates would be counted as run-to-run variance in σ̂. The two
# zero-decay intervention arms took it to 42 and 41 on 2026-10-19 (views-models#568);
# their requirements.txt files are byte-identical copies of their seed-matched controls'.
DEFERRED_PACKAGES: dict[str, str] = {
    "views-datafactory": (
        "#509: the three legs that run on rented hardware are floored at >=1.13.0 for the "
        "credential-handling fixes (both postprocessor requirement files and the pod install); "
        "the 41 model declarations stay at >=1.9.0 because they are unrelated to the delivery "
        "and one is violet_visitor, owned by another session. "
        "TRIGGER: #509 raising the remaining 41 declarations to >=1.13.0 — at which point this "
        "entry is deleted, not amended, because the divergence it describes no longer exists."
    ),
}

# ⛔ `views-pipeline-core` was the second entry above until #635, and it is gone rather than
# amended, which is what its own TRIGGER instructed. The divergence it described — nineteen
# declarations at >=3.2.0 against twelve ensembles at >=3.0.0 — no longer exists: all 31 are
# at >=3.2.0,<4.0.0, so `test_a_package_is_declared_the_same_way_everywhere` now covers the
# package under the general rule and needs no exemption.
#
# The FLOOR did not stop mattering when the divergence did. #594's trap is a property of
# every one of those sources, not of the gap between them: each carries
# configs/config_maturity.py, which only pipeline-core >=3.2.0 can find. That property is now
# asserted directly by test_every_pipeline_core_declaration_is_at_the_3_2_0_floor below,
# which is the guard the deleted entry's reason paragraph used to carry in prose.

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


def test_every_pipeline_core_declaration_keeps_its_upper_bound():
    """Kept deliberately after #635 deleted the deferral it used to compensate for.

    While `views-pipeline-core` sat in DEFERRED_PACKAGES it was exempt from
    `test_no_dependency_is_declared_without_an_upper_bound`, and this narrower rule was the
    only thing that would notice `<4.0.0` being dropped from a pipeline-core line — a major
    release would otherwise install itself on the next monthly run. #635 removed that entry,
    so the general rule covers the package again and this one is now redundant.

    ⛔ It stays anyway, because the redundancy is conditional on an absence: re-adding a
    `views-pipeline-core` entry to DEFERRED_PACKAGES for any future reason silently reopens
    the hole, and whoever adds it will be thinking about floors, not ceilings. A guard that
    costs eight lines and survives the exemption is cheaper than rediscovering why it was
    written. `views-datafactory`'s entry documents the same pattern for itself.
    """
    unbounded = []
    for name, number, req in _declarations():
        if req.name != "views-pipeline-core" or req.url:
            continue
        if not any(s.operator in ("<", "<=", "==", "===", "~=") for s in req.specifier):
            unbounded.append(f"  {name}:{number}: {req}")
    assert not unbounded, (
        "views-pipeline-core declared without a ceiling, and its DEFERRED_PACKAGES entry "
        "means the general upper-bound rule skips it:\n" + "\n".join(unbounded)
    )


def test_every_pipeline_core_declaration_is_at_the_3_2_0_floor():
    """#594's trap, asserted as a property of each source rather than of a divergence.

    Every source that declares `views-pipeline-core` carries `configs/config_maturity.py`,
    and only pipeline-core >=3.2.0 can find that file; below it the loader looks for a
    `configs/config_deployment.py` none of them has. views-baseline 1.0.2 floors the package
    at >=3.0.0, so an environment holding exactly 3.0.0 satisfies `>=3.0.0`, the launcher
    upgrades nothing, and the run dies on the missing file. That is what broke Hop A on
    2026-10-22 (#594). #635 brought the last twelve ensembles up, so the floor is now uniform
    and this test replaces `test_the_pipeline_core_deferral_still_describes_reality`.

    ⛔ **The thirty-one are pinned BY NAME, because a count cannot survive a compensating
    swap.** That is not a hypothesis: under the predecessor of this test, dropping
    `models/ashen_sparrow` to `>=3.0.0` **and** raising `ensembles/skinny_love` to `>=3.2.0`
    held both counts at 12 and 19 exactly, left the whole suite green, and put a load-bearing
    member on the floor its own requirements.txt calls "what broke Hop A" — while the test
    reported that the deferral "still describes reality" (C-182).

    Keying the census by specifier was tried first and does NOT close it: it pins which
    *specifier* holds which count, while the swap moves which *file* holds which floor, and
    the counts are unchanged either way. Only the identities close it. A fix for a surviving
    mutation is a hypothesis until the mutation is re-run against it.

    Two assertions, and neither is redundant. The name set catches a source leaving the
    census or joining it unreviewed — a *smaller* census reports fewer passes, not a failure,
    which is C-182's other half. The single-key census catches a declaration drifting to a
    different specifier, and a second pipeline-core line appearing inside one file.
    """
    #: Every source that declares views-pipeline-core directly: epic #595's sixteen members
    #: and all fifteen ensembles. Each can read configs/config_maturity.py only at >=3.2.0.
    #: A new source declaring the package belongs here as a reviewed edit, not silently.
    DECLARES_PIPELINE_CORE = {f"models/{stem}_{suffix}"
                              for stem in ("ashen", "brisk", "candid", "dusky",
                                           "eager", "faint", "gentle", "humble")
                              for suffix in ("sparrow", "condor")} | {
        f"ensembles/{name}" for name in (
            "chrome_box", "chrome_crate", "chunky_bunny", "cruel_summer", "first_love",
            "golden_hour", "pink_ponyclub", "rude_boy", "rusty_bucket", "skinny_love",
            "stellar_horizon", "synthetic_chant", "synthetic_choir", "synthetic_chorus",
            "white_mustang",
        )
    }

    floors: dict[str, int] = {}
    declaring = set()
    for name, _number, req in _declarations():
        if req.name != "views-pipeline-core" or req.url:
            continue
        floors[str(req.specifier)] = floors.get(str(req.specifier), 0) + 1
        declaring.add(str(Path(name).parent))

    assert declaring == DECLARES_PIPELINE_CORE, (
        "the sources declaring views-pipeline-core are not the thirty-one named here.\n"
        f"  stopped declaring it : {sorted(DECLARES_PIPELINE_CORE - declaring)}\n"
        f"  started unreviewed   : {sorted(declaring - DECLARES_PIPELINE_CORE)}\n"
        "A source that stops declaring the package inherits views-baseline's >=3.0.0 floor "
        "and carries #594's trap again (C-116). A new one needs the floor and this name."
    )
    # `str(SpecifierSet)` normalises the clause order, hence `<4.0.0,>=3.2.0`.
    assert floors == {"<4.0.0,>=3.2.0": 31}, (
        "views-pipeline-core is no longer declared as >=3.2.0,<4.0.0 by all thirty-one "
        f"sources: {floors}. Anything below >=3.2.0 cannot find configs/config_maturity.py "
        "and dies on a missing config_deployment.py (#594)."
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


def test_the_deferral_trigger_counts_the_declarations_it_actually_describes():
    """The number in the #509 trigger is load-bearing, so it is measured, not typed.

    `DEFERRED_PACKAGES["views-datafactory"]` says "the N model declarations stay at >=1.9.0"
    and "TRIGGER: #509 raising the remaining N". Whoever acts on that trigger floors N
    declarations and deletes the entry — and the entry also suppresses the upper-bound rule,
    so a declaration left behind loses two protections at once while the suite stays green.

    The count said 34 and was correct until a new model directory arrived carrying its own
    `views-datafactory>=1.9.0`. Nothing noticed, because prose does not fail.
    """
    def declares_unfloored(path: Path) -> bool:
        # Parsed per line with comments stripped. Matching the raw text counted
        # postprocessors/un_fao/requirements.txt, which is floored at >=1.13.0 on its
        # requirement line and mentions >=1.9.0 twice in comments explaining the history.
        # A guard that reads prose about the code instead of the code reports the wrong
        # number with total confidence.
        for line in path.read_text(encoding="utf-8").splitlines():
            spec = line.split("#", 1)[0].strip()
            if spec.startswith("views-datafactory") and ">=1.9.0" in spec:
                return True
        return False

    unfloored = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in REPO_ROOT.glob("*/*/requirements.txt")
        if declares_unfloored(path)
    )
    note = DEFERRED_PACKAGES["views-datafactory"]
    claimed = {int(m) for m in re.findall(r"(?:the|remaining) (\d+)", note)}
    assert claimed == {len(unfloored)}, (
        f"the views-datafactory deferral claims {sorted(claimed)} unfloored model "
        f"declaration(s); there are {len(unfloored)}:\n  "
        + "\n  ".join(unfloored)
        + "\n\nUpdate the note, in BOTH sentences. Someone will act on this trigger by "
        "flooring exactly that many files and deleting the entry; a file left behind loses "
        "the upper-bound rule too, because this entry suppresses it."
    )
