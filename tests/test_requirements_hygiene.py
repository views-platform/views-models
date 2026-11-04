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
from packaging.utils import canonicalize_name
from packaging.version import Version

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
# 41 -> 45 on 2026-11-02: EXP-02's four 300-lesson arms (views-models#685), byte-identical
# to each other at >=1.9.0. ⚠️ This number counts the FILESYSTEM, so it reads 47 on a working
# tree that still holds the two untracked `*tinyzero*` experiment dirs. 45 is the fresh-clone
# truth and the one CI measures; a local run showing 47 is those two dirs, not a regression.
# 45 -> 53 on 2026-11-03: EXP-03's eight arms, seeds 44-47, byte-identical requirements.txt to
# the EXP-02 four. A working tree still holding the two `*tinyzero*` dirs reads 55, not 53.
DEFERRED_PACKAGES: dict[str, str] = {
    "views-datafactory": (
        "#509: the three legs that run on rented hardware are floored at >=1.13.0 for the "
        "credential-handling fixes (both postprocessor requirement files and the pod install); "
        "the 53 model declarations stay at >=1.9.0 because they are unrelated to the delivery "
        "and one is violet_visitor, owned by another session. "
        "TRIGGER: #509 raising the remaining 53 declarations to >=1.13.0 — at which point this "
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
# asserted directly by test_every_pipeline_core_declaration_is_at_the_floor_its_source_needs below,
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

    ⛔ **Correction, from the independent audit of #635: this rule has never fired alone and
    cannot today.** An earlier draft of this docstring claimed it "was the only thing that
    would notice `<4.0.0` being dropped" under a re-added exemption. That was measured and is
    false: with a `views-pipeline-core` entry back in DEFERRED_PACKAGES *and* the ceiling
    dropped from one declaration, both this test and
    `test_every_pipeline_core_declaration_is_at_the_floor_its_source_needs` go red, because dropping the
    ceiling changes `str(req.specifier)` away from `<4.0.0,>=3.2.0` and that guard pins the
    whole specifier. There is no mutation this catches and that one misses.

    It is kept, and the honest reason is narrow: it is the only *semantic* statement about the
    ceiling, so it survives the floor guard's specifier pin being loosened into a version
    comparison — which is a change someone will reasonably propose, since the pin also reddens
    a uniform bump to `==3.3.0`. Eight lines against that is worth it. What it must not be
    credited with is independent coverage today. `views-datafactory`'s entry documents the
    same exemption-widens-silently pattern for itself.
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


def test_every_pipeline_core_declaration_is_at_the_floor_its_source_needs():
    """#594's trap, asserted as a property of each source rather than of a divergence.

    Every source that declares `views-pipeline-core` carries `configs/config_maturity.py`,
    and only pipeline-core >=3.2.0 can find that file; below it the loader looks for a
    `configs/config_deployment.py` none of them has. views-baseline 1.0.2 floors the package
    at >=3.0.0, so an environment holding exactly 3.0.0 satisfies `>=3.0.0`, the launcher
    upgrades nothing, and the run dies on the missing file. That is what broke Hop A on
    2026-10-22 (#594). #635 brought the last twelve ensembles up, and this test replaces
    `test_the_pipeline_core_deferral_still_describes_reality`.

    ⛔ **The floor is no longer uniform, and that is the point of `NEEDS_A_HIGHER_FLOOR`.**
    `ensembles/chrome_box` sits at >=3.3.5 because it is the only source that reaches
    pipeline-core's `load_cm_frame` (it declares `reconciliation: "pgm_cm"`), and before
    3.3.5 that function reported a cm forecast as absent when its partner wrote a
    PredictionFrame directory rather than a parquet (views-pipeline-core#562). So each
    source is now compared against **the floor it needs**, not against one shared string.
    That is strictly stronger than the uniform check it replaces: a uniform assertion
    passes a source that was raised without review, and this one does not.

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

    Three assertions, and none is redundant. The name set catches a source leaving the
    census or joining it unreviewed — a *smaller* census reports fewer passes, not a failure,
    which is C-182's other half. The per-source comparison catches any declaration differing
    from the floor its own directory needs, in EITHER direction. The specifier census catches
    a second pipeline-core line appearing inside one file, which the per-source check cannot
    see because it would compare each line separately and find both acceptable.
    """
    #: Every source that declares views-pipeline-core directly: epic #595's sixteen members
    #: and all fifteen ensembles. Each can read configs/config_maturity.py only at >=3.2.0.
    #: A new source declaring the package belongs here as a reviewed edit, not silently.
    DECLARES_PIPELINE_CORE = {f"models/{stem}_{suffix}"
                              for stem in ("ashen", "brisk", "candid", "dusky",
                                           "eager", "faint", "gentle", "humble")
                              for suffix in ("sparrow", "condor")} | {
        #: The ten views-models#695 smoothed climatology arms. They declare the package for
        #: the same reason the sixteen above do: views-baseline's own floor is >=3.0.0, which
        #: is BELOW the 3.2.0 that can find configs/config_maturity.py, so an environment
        #: already holding exactly 3.0.0 satisfies it and the launcher upgrades nothing —
        #: then the run dies on a missing config_deployment.py. That is #594, and inheriting
        #: the lower floor is how it recurs (C-116).
        f"models/{name}" for name in (
            "short_linen", "plain_linen", "long_linen",
            "sheer_velvet", "plush_velvet", "flat_velvet",
            "light_tweed", "even_tweed", "heavy_tweed",
            "plain_burlap",
        )} | {
        f"ensembles/{name}" for name in (
            "chrome_box", "chrome_crate", "chunky_bunny", "cruel_summer", "first_love",
            "golden_hour", "pink_ponyclub", "rude_boy", "rusty_bucket", "skinny_love",
            "stellar_horizon", "synthetic_chant", "synthetic_choir", "synthetic_chorus",
            "white_mustang",
        )
    }

    #: ⛔ The one source that needs MORE than the shared floor, and why. Adding an entry
    #: here is a reviewed edit: it must name a capability the source actually calls, not a
    #: version someone happened to have installed. Removing one re-admits the defect.
    BASE_FLOOR = "<4.0.0,>=3.2.0"
    NEEDS_A_HIGHER_FLOOR = {
        #: Two capabilities stack here, and the HIGHER one is what the floor satisfies.
        #: >=3.3.5 — calls through to `load_cm_frame`, which could not read a PredictionFrame
        #:   cm forecast before it (views-pipeline-core#562, published 2026-10-29).
        #: >=3.4.0 — ⛔ and this is not about pipeline-core at all: 3.4.0 is the first release
        #:   carrying `views-frames>=3.0.0`, and views-frames 3.0.0 is where the reconciler was
        #:   fixed. Below it a country whose grid is all-zero in a draw loses its entire mass
        #:   silently — 4.023%/7.321%/2.943% on the 2026-10-30 delivery, which reported
        #:   STATUS: OK and scaling_failures: 0 (views-frames#283, #659/#663). Raised
        #:   2026-11-02, the day views-datafactory 1.17.0 made it satisfiable at all.
        "ensembles/chrome_box": "<4.0.0,>=3.4.0",
    }
    assert set(NEEDS_A_HIGHER_FLOOR) <= DECLARES_PIPELINE_CORE, (
        "a source is excused to a higher floor without appearing in the census at all, so "
        f"nothing checks it: {sorted(set(NEEDS_A_HIGHER_FLOOR) - DECLARES_PIPELINE_CORE)}"
    )

    floors: dict[str, int] = {}
    declaring = set()
    offenders: list[str] = []
    inert: list[str] = []
    for name, number, req in _declarations():
        #: ⛔ `packaging` does NOT canonicalise `Requirement.name`, so `views_pipeline_core`
        #: is a different string to this guard and the SAME package to pip. An independent
        #: audit of this guard appended `views_pipeline_core==3.1.0` to a co-tenant and the
        #: whole suite stayed green while the shared prefix resolved below the floor.
        if canonicalize_name(req.name) != "views-pipeline-core":
            continue
        #: ⛔ A URL, a marker, or a file that is not actually named `requirements.txt`
        #: each change WHAT INSTALLS while leaving the specifier string intact. None of them
        #: may `continue` past the census: a skip drops the directory out of `declaring`, and
        #: a smaller census reports fewer passes rather than failing (C-182). They are
        #: collected and asserted instead, so the guard refuses rather than shrinks.
        if req.url:
            inert.append(f"  {name}:{number}: declared as a URL, so the floor is a label: {req}")
        elif req.marker is not None:
            inert.append(f"  {name}:{number}: carries an environment marker, so this line "
                         f"may install nothing: {req}")
        elif Path(name).name != "requirements.txt":
            inert.append(f"  {name}:{number}: not named requirements.txt, so pip never reads "
                         f"it: {req}")
        else:
            source = str(Path(name).parent)
            floors[str(req.specifier)] = floors.get(str(req.specifier), 0) + 1
            declaring.add(source)
            expected = NEEDS_A_HIGHER_FLOOR.get(source, BASE_FLOOR)
            if str(req.specifier) != expected:
                offenders.append(
                    f"  {name}:{number}: declares {req.specifier}, needs {expected}"
                )

    assert not inert, (
        "a views-pipeline-core declaration cannot be read as a floor at all, so this guard "
        "refuses rather than counting it (ADR-025):\n" + "\n".join(inert)
    )
    assert declaring == DECLARES_PIPELINE_CORE, (
        "the sources declaring views-pipeline-core are not the thirty-one named here.\n"
        f"  stopped declaring it : {sorted(DECLARES_PIPELINE_CORE - declaring)}\n"
        f"  started unreviewed   : {sorted(declaring - DECLARES_PIPELINE_CORE)}\n"
        "A source that stops declaring the package inherits views-baseline's >=3.0.0 floor "
        "and carries #594's trap again (C-116). A new one needs the floor and this name."
    )
    assert not offenders, (
        "a views-pipeline-core declaration is not at the floor its own source needs. "
        "Anything below >=3.2.0 cannot find configs/config_maturity.py and dies on a "
        "missing config_deployment.py (#594); chrome_box additionally cannot read "
        "chrome_crate's cm forecast below >=3.3.5 (#562). A source raised ABOVE its "
        "floor is also listed, because an unreviewed raise is how a floor stops meaning "
        "anything:\n" + "\n".join(offenders)
    )
    # `str(SpecifierSet)` normalises the clause order, hence `<4.0.0,>=3.2.0`.
    expected_census = {BASE_FLOOR: len(DECLARES_PIPELINE_CORE) - len(NEEDS_A_HIGHER_FLOOR)}
    for spec in NEEDS_A_HIGHER_FLOOR.values():
        expected_census[spec] = expected_census.get(spec, 0) + 1
    assert floors == expected_census, (
        f"the views-pipeline-core specifier census is {floors}, not {expected_census}. "
        "The per-source check above passed, so every LINE is at its source's floor — this "
        "catches the case it cannot see: one file carrying two pipeline-core lines, where "
        "each line is individually acceptable and pip resolves their intersection."
    )


def test_chrome_box_refuses_the_pipeline_core_that_cannot_read_a_pf_cm_forecast():
    """The floor stated as the PROPERTY it is for, not as the string it happens to be.

    The census above pins `ensembles/chrome_box` to the exact specifier, which is what
    catches an unreviewed edit. This asserts the thing that specifier is *for*, so the
    reason survives the next legitimate bump: whatever the floor says, it must exclude
    3.3.4 and admit 3.3.5.

    **Why 3.3.4 by name.** `chrome_box` declares `reconciliation: "pgm_cm"`, so
    PredictionFrameEnsembleManager calls `load_cm_frame` to read `chrome_crate`'s cm
    forecast. Up to and including 3.3.4 that function looked only for a `.parquet` file
    while a PredictionFrame partner writes a directory, so it reported
    "no local forecast found for country model 'chrome_crate'" over a forecast that
    existed (views-pipeline-core#562, fixed in 3.3.5, ADR-069).

    ⛔ **Both directions, because one is vacuous alone.** A specifier of `>=99.0` excludes
    3.3.4 and would satisfy a one-sided test while installing nothing that exists. The
    admit-3.3.5 half is what makes the refuse-3.3.4 half mean something.
    """
    specs = [
        req.specifier
        for name, _number, req in _declarations()
        if canonicalize_name(req.name) == "views-pipeline-core"
        and str(Path(name).parent) == "ensembles/chrome_box"
    ]
    assert len(specs) == 1, (
        f"ensembles/chrome_box declares views-pipeline-core {len(specs)} times, so this "
        "guard cannot say which floor applies and refuses rather than reading the first "
        "one (ADR-025). pip would resolve the intersection of all of them."
    )
    spec = specs[0]
    assert Version("3.3.4") not in spec, (
        f"ensembles/chrome_box admits views-pipeline-core 3.3.4 ({spec}), which cannot "
        "read chrome_crate's PredictionFrame cm forecast and reports it as absent "
        "(views-pipeline-core#562). A laptop whose shared envs/views_ensemble already "
        "holds 3.3.4 satisfies the range, so run.sh upgrades nothing and the run dies."
    )
    #: ⛔ 3.3.5 IS NOW ALSO REFUSED, and it is the more interesting of the two.
    #: 3.3.5 carries the #562 fix, so it starts and finishes and writes a pool. What it does
    #: not carry is `views-frames>=3.0.0`, so it reconciles with the OLD reconciler: a country
    #: whose grid is all-zero in a draw loses its whole mass, with no error, no warning, and
    #: `scaling_failures: 0`. That is the 2026-10-30 delivery, which was short by
    #: 4.023%/7.321%/2.943% and reported STATUS: OK (views-frames#283).
    #: ⇒ So 3.3.4 fails LOUDLY and 3.3.5 fails SILENTLY, and the silent one is why the floor
    #: moved. Asserting only the 3.3.4 half would now pass on the version that loses mass.
    assert Version("3.3.5") not in spec, (
        f"ensembles/chrome_box admits views-pipeline-core 3.3.5 ({spec}). It runs to "
        "completion and silently drops 2.9-7.3% of the fatality mass, because it does not "
        "floor views-frames at 3.0.0 (views-frames#283, #659/#663). Unlike 3.3.4 this one "
        "leaves no trace in the run."
    )
    assert Version("3.4.0") in spec, (
        f"ensembles/chrome_box no longer admits views-pipeline-core 3.4.0 ({spec}), the "
        "first release carrying the views-frames>=3.0.0 floor that fixes the reconciler. A "
        "floor that excludes every version carrying the fix refuses the broken ones for free "
        "and installs nothing."
    )

def test_every_chrome_box_member_floors_the_pins_that_cap_views_frames():
    """⛔ The eight members pinned BY NAME, because a set cannot see a swap.

    **This guard exists because its absence was measured, not imagined.** After the floors
    were raised, a post-commit mutation pass reverted `models/ashen_sparrow` to
    `views-baseline>=1.0.2` and `models/brisk_sparrow` to `views-datafactory>=1.13.0` — the
    two pins that CAP `views-frames` below 3 — and **the whole suite stayed green**. Both
    mutations survived.

    Why the existing checks could not see it, which is the part worth keeping:
    `test_a_package_is_declared_the_same_way_everywhere` compares the **set** of specifiers
    against `EXPECTED_DIVERGENCE`. Reverting one of eight files leaves that set unchanged —
    `{>=1.0.2, >=1.0.3}` either way — because seven siblings still hold the raised floor. It
    pins which *specifier* coexists with which, while a swap moves which *file* holds which
    floor. That is **C-182** exactly, the compensating swap that held two counts at 12 and 19
    while putting a load-bearing member on the floor its own requirements.txt calls "what
    broke Hop A". The pipeline-core census answered it by naming identities; so does this.

    **What the pins are for.** These eight are `chrome_box`'s members, and `chrome_box`
    reconciles. `views-baseline 1.0.2` and every `views-datafactory` up to 1.16.0 declare
    `views-frames<2` as a BASE dependency, so either one makes views-frames 3.x unreachable
    for anything sharing the environment — and below 3.0.0 a country whose grid is all-zero
    in a draw loses its entire mass silently (views-frames#283, C-192). A cap refuses; it
    does not merely prefer.

    ⚠️ `views-pipeline-core` is deliberately NOT checked here. These members do not
    reconcile, so the `>=3.4.0` floor is not theirs to carry — only the two pins that cap
    frames are.
    """
    #: The eight by name. A member leaving this set is a reviewed edit, not a silent one,
    #: and a SMALLER census reports fewer passes rather than failing — which is C-182's
    #: other half, closed by the coverage assertion below.
    MEMBERS = {f"models/{stem}_sparrow" for stem in (
        "ashen", "brisk", "candid", "dusky", "eager", "faint", "gentle", "humble")}
    #: package -> (floor, the cap it exists to escape)
    REQUIRED = {
        "views-baseline": ((1, 0, 3), "1.0.2 declares views-frames<2"),
        "views-datafactory": ((1, 17, 0), "every release up to 1.16.0 declares views-frames<2"),
    }

    found: dict[str, dict[str, tuple[int, int, int]]] = {}
    for name, _number, req in _declarations():
        source = str(Path(name).parent)
        if source not in MEMBERS or Path(name).name != "requirements.txt":
            continue
        pkg = canonicalize_name(req.name)
        if pkg not in REQUIRED:
            continue
        #: ⛔ A URL or a marker changes WHAT INSTALLS while leaving the specifier readable,
        #: so neither may `continue` past the census (ADR-025) — record an unreadable floor
        #: as absent, and the coverage assertion below turns that into a failure.
        if req.url or req.marker is not None:
            continue
        m = re.search(r">=\s*(\d+)\.(\d+)\.(\d+)", str(req.specifier))
        if m:
            found.setdefault(source, {})[pkg] = tuple(int(g) for g in m.groups())

    missing = sorted(MEMBERS - set(found))
    assert not missing, (
        f"these chrome_box members declare no readable floor for {sorted(REQUIRED)} at all, "
        f"so this guard examined nothing for them and refuses rather than passing "
        f"(ADR-025): {missing}"
    )
    offenders: list[str] = []
    for source in sorted(MEMBERS):
        for pkg, (floor, why) in REQUIRED.items():
            got = found[source].get(pkg)
            if got is None:
                offenders.append(f"  {source}: declares no {pkg} floor")
            elif got < floor:
                offenders.append(
                    f"  {source}: {pkg} floored at "
                    f"{'.'.join(str(n) for n in got)}, needs "
                    f"{'.'.join(str(n) for n in floor)} — {why}"
                )
    assert not offenders, (
        "a chrome_box member admits a release that caps views-frames below 3, which makes "
        "the fixed reconciler unreachable for the environment it shares. Below views-frames "
        "3.0.0 a country whose grid is all-zero in a draw loses its whole mass with no "
        "error (views-frames#283, C-192):\n" + "\n".join(offenders)
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

    ⛔ **One divergence is expected and is declared below rather than exempted.** The
    distinction this rule actually cares about is whether run order can change the outcome,
    and for `views-pipeline-core` today it cannot, because the two floors are **nested**:

        ensembles/chrome_crate  >=3.2.0,<4.0.0     ─┐ same ceiling,
        ensembles/chrome_box    >=3.3.5,<4.0.0     ─┘ one floor strictly higher

    Both run in the shared `envs/views_ensemble`, and pip does not downgrade a package to
    satisfy a floor already met. So `chrome_crate` first leaves 3.3.4 in place and
    `chrome_box` then upgrades to 3.3.5; `chrome_box` first installs 3.3.5 and
    `chrome_crate` accepts it. **Neither order loses** — which is exactly what makes this
    divergence different from the kind C-116 describes, where the specs are incompatible and
    the loser is whoever ran first.

    ⚠️ This is NOT handled with a `DEFERRED_PACKAGES` entry, although the assertion message
    below offers that. Two reasons. That list means *a decision was deliberately deferred*,
    and this divergence is a decided, permanent, reasoned difference — mislabelling it there
    would make the list a place to hide, which its own docstring warns about at exactly two
    entries. And an entry is blanket: it would stop this rule seeing **every** pipeline-core
    declaration, including a genuinely incompatible one appearing later. The dict below is
    narrower — a third specifier, a changed specifier, or any other package diverging still
    fails.

    The specs are pinned as STRINGS, following the same deliberate choice the floor census
    makes: a version-comparison form would accept an unreviewed bump, and a reviewed bump is
    the point. Updating chrome_box's floor means editing two lines in this file, on purpose.
    """
    #: package -> the exact set of specifiers expected to coexist, and why order cannot
    #: decide the outcome. Anything not listed here must be declared identically everywhere.
    EXPECTED_DIVERGENCE: dict[str, dict[str, str]] = {
        "views-pipeline-core": {
            "<4.0.0,>=3.2.0": "every source: configs/config_maturity.py needs >=3.2.0 (#594)",
            "<4.0.0,>=3.4.0": "ensembles/chrome_box only: the only source that reconciles, "
                              "so the only one needing the views-frames>=3.0.0 floor that "
                              "3.4.0 carries (views-frames#283, #663)",
        },
        #: Raised on the eight chrome_box members only, 2026-11-02. views-baseline 1.0.2
        #: declares `views-frames<2`, which makes views-frames 3.x UNREACHABLE for anything
        #: sharing the environment — a cap refuses, it does not merely prefer. 1.0.3 widened
        #: it to `<4`. The nesting argument above applies unchanged: same ceiling, one floor
        #: strictly higher, and pip does not downgrade to satisfy a floor already met, so
        #: neither run order loses. The eight `*_condor` members stay at >=1.0.2 because they
        #: install into a different prefix (envs/views-baseline) and do not reconcile —
        #: claiming a requirement they do not have is the false precision this file argues
        #: against elsewhere.
        #: Raised on the six UpsampleConv2 arms only, 2026-11-04. `UpsampleConv2` first ships
        #: in views-hydranet **0.1.8** — verified by downloading 0.1.4 through 0.1.9 from PyPI and
        #: grepping each wheel, not by reading a changelog. The other 21 HydraNets declare
        #: `HydraBNUNet06_LSTM4`, which every 0.1.x provides, so raising their floor would claim a
        #: requirement they do not have — the false precision this file argues against elsewhere.
        #:
        #: ⛔ The defect this closes: `~=0.1.1` is `>=0.1.1,<0.2.0`, so a config naming
        #: `UpsampleConv2` could resolve 0.1.7 and find the architecture absent from
        #: `architectures/registry.py`. The six arms ran green only because a fresh install picks
        #: the latest (0.1.9); a REUSED venv, or `envs/views-hydranet` when it held 0.1.5, would
        #: not have it. The floor does not change what a fresh resolve installs — it stops the
        #: config silently accepting a version that cannot honour it.
        #:
        #: The nesting argument above applies unchanged: same ceiling `<0.2.0`, one floor strictly
        #: higher, one shared prefix, and pip does not downgrade to satisfy a floor already met.
        #: So neither run order loses.
        "views-hydranet": {
            "~=0.1.1": "the 21 HydraNets on HydraBNUNet06_LSTM4: any 0.1.x provides it",
            "<0.2.0,>=0.1.8": "the six UpsampleConv2 arms only: 0.1.8 is the first release "
                              "carrying that architecture (views-hydranet#491/#546)",
        },
        "views-baseline": {
            "<2.0.0,>=1.0.2": "every other source: does not reconcile, so no frames floor",
            "<2.0.0,>=1.0.3": "the eight models/*_sparrow (chrome_box's members): 1.0.2's "
                              "`views-frames<2` cap would block the fixed reconciler",
            # ⛔ A CAPABILITY FLOOR, not a cap-avoidance one — which is why it is higher than
            # either of the above and is not an inconsistency with them. 1.1.0 is the first
            # release carrying the smoothed climatology algorithms (views-baseline ADR-023):
            # InterpolatedQuantileConflictology, KDEConflictology and
            # EmpiricalKDEMixtureConflictology. On 1.0.x these ten arms name a class the
            # registry cannot build, so the floor is the difference between running and not.
            # The nesting argument holds as it does above: same ceiling, one floor strictly
            # higher, and pip does not downgrade to satisfy a floor already met.
            "<2.0.0,>=1.1.0": "the ten views-models#695 smoothed climatology arms: 1.1.0 is "
                              "the first release that has the algorithms they name",
        },
    }

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

    divergent = {
        pkg: v for pkg, v in specs.items()
        if len(v) > 1 and set(v) != set(EXPECTED_DIVERGENCE.get(pkg, {}))
    }
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
    #: ⛔ The other direction, which an allowlist always forgets: an entry describing a
    #: divergence that no longer exists is a live exemption nobody is checking. #635 deleted
    #: pipeline-core's DEFERRED_PACKAGES entry rather than amending it for exactly this
    #: reason, and this assertion is what makes the same discipline automatic here.
    stale = {
        pkg: sorted(set(expected) - set(specs.get(pkg, {})))
        for pkg, expected in EXPECTED_DIVERGENCE.items()
        if set(expected) - set(specs.get(pkg, {}))
    }
    assert not stale, (
        "EXPECTED_DIVERGENCE describes specifiers that no source declares any more, so the "
        f"exemption is wider than the divergence it was written for: {stale}\n"
        "Delete the entry rather than amending it — if the divergence is gone, the general "
        "rule should cover the package again."
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
