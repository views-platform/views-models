"""`chrome_box`'s config must keep the MEASURED peak, not a derived floor.

**Why this guard and not the one that was here before.** `tests/test_falsification_crafd_chain_floors.py`
asserted `MEM_FLOOR_GB >= 68` against `tools/podrun/pod_run_crafd_chain.sh`. That was
wrong, and it was my own error: 68.2 GiB is `chrome_box`'s **pgm** pool peak, while that
runner executes only `chrome_crate` (**cm**, phase 1 of 2 — its header says so, and its
manifest records `pool: 96M`, members ~2 MB RSS, 7 minutes, completed). A pgm peak used
as a cm floor is a correct value answering the wrong question, and the floor was never
too low.

⛔ Earlier versions of this docstring said "views-models#648 was filed on that basis and
is invalid". **No such issue had been filed** — 647 was the highest number the repository
had, and I had invented 648. Corrected before this file was ever pushed. (#648 now exists
and is the PR carrying this correction: GitHub numbers issues and pull requests from one
counter, so opening it consumed the number. It is not, and never was, an issue about a
memory floor.) It is
recorded here rather than silently deleted because a fabricated issue number is the most
citable kind of wrong: it reads as provenance, and the next reader would have gone looking
for a decision that was never taken.

The real exposure is the opposite shape: phase 2 is **not scripted**, so no preflight
guards it at all, and the only thing standing between an operator and a box that dies
after ~3 hours of paid member training is the comment in
`ensembles/chrome_box/configs/config_hyperparameters.py`. A comment is the weakest kind
of protection, so it is the one worth pinning — especially this one, which has already
been wrong three times (19.8 GiB from `memory.current`, a 27.8 GiB undersample, and a
"1.9x" ratio whose floor counted one pooled aggregate instead of three).

⛔ This asserts the NUMBERS and their RELATIONSHIP, not the prose around them. A guard
that matched sentences would fail on every rewording and teach the next person to delete
it.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "ensembles" / "chrome_box" / "configs" / "config_hyperparameters.py"
CRATE_CONFIG = REPO / "ensembles" / "chrome_crate" / "configs" / "config_hyperparameters.py"

#: Measured on the pod, 2026-10-28: 30s sampling of cgroup anon through pool + reconcile.
#:
#: ⛔ WHICH RUN, because the evidence dir holds two and only one of them reconciled.
#: `phase2.out` (15:25) is the UNPATCHED run: it pooled, then died three times with
#: "Cannot reconcile: no local forecast found for country model 'chrome_crate'" — that is
#: views-pipeline-core#562 live, and its pool reached only 27.8 GiB (the superseded figure
#: below). 68.20 GiB comes from `probe.out` (16:53-16:58), run with `probe_sitecustomize.py`
#: (named `sitecustomize.py` on the pod, which is how it loaded)
#: monkeypatching `load_cm_frame` to prefer the PredictionFrame directory — functionally
#: what 3.3.5 now does natively. That run reconciled all three targets
#: (`mode=aligned-draws, 2330712 grid rows, 1024 draws`, three "Reconciled ... (pgm_cm)"
#: lines) and finished: "Done. Runtime: 4.427 minutes". So the peak DOES cover reconcile.
#: Sampled by `watch_probe.sh`, which watches probe.out specifically — not by
#: `anon_sampler.sh`, which is the 10s phase-2 sampler and watches the run that refused.
#:
#: ⚠️ It is a LOWER BOUND, for two reasons worth keeping: 30s sampling over a 4.4-minute
#: stage is ~9 samples, so a shorter transient is invisible; and only the watcher's final
#: `PEAK anon:` line was transcribed, not the series — the pod is terminated, so it cannot
#: be recovered. Treat 68.20 as "at least", and the 1.27x over the frame floor likewise.
MEASURED_PEAK_GIB = 68.20
#: Frame bytes alone, from the real array shapes. Deliberately NOT the sizing input.
FRAME_FLOOR_GIB = 53.86


def _config_source() -> str:
    assert CONFIG.exists(), (
        f"{CONFIG.relative_to(REPO)} is gone. It carries the only memory figure for a "
        f"phase that has no preflight script; this guard cannot see its subject (ADR-025)."
    )
    return CONFIG.read_text(encoding="utf-8")


def _gib_figures(src: str) -> list[float]:
    """Every `<number> GiB` the config states, in order."""
    return [float(m) for m in re.findall(r"(\d+\.?\d*)\s*GiB", src)]


def _peak_claims(src: str) -> dict[str, float]:
    """Every figure the config LABELS as the measured peak, by where it says it.

    ⛔ Why labelled claims and not "does 68.2 appear anywhere". The first version of
    this guard asked whether the file contained the peak and whether its largest GiB
    figure was at least the peak. Two mutations walked straight through: rewriting the
    table row `68.20 GiB` to the cheaper frame floor, and softening the headline to the
    superseded `27.8 GiB`. Both survived because the OTHER statement still carried the
    right number, and because the largest figure in the file is the 87 GiB box class,
    which satisfies any `>=` test for free.

    The config states the peak in three places on purpose — a headline, a table row and
    the subtraction that bounds the unattributed 14.3 GiB. Three statements of one
    measurement is a redundancy worth having, but only if they are required to AGREE.
    Softening one of three is precisely how a figure gets quietly lowered.
    """
    claims: dict[str, float] = {}
    patterns = {
        "headline": r"PEAK IS\s*(\d+\.?\d*)\s*GiB",
        "table row": r"measured peak[^\n]*?(\d+\.?\d*)\s*GiB",
        "subtraction": r"(\d+\.?\d*)\s*measured\s*\n?\s*minus",
    }
    for where, pattern in patterns.items():
        found = re.findall(pattern, src)
        assert found, (
            f"the config no longer states the measured peak as a {where}; this guard "
            f"cannot see that part of its subject, so it refuses rather than passing "
            f"on the strength of the other two (ADR-025)."
        )
        assert len(set(found)) == 1, (
            f"the {where} states the peak more than once and inconsistently: {found}"
        )
        claims[where] = float(found[0])
    return claims


def test_the_measured_peak_is_stated_in_the_config() -> None:
    claims = _peak_claims(_config_source())
    disagreeing = {w: v for w, v in claims.items() if abs(v - MEASURED_PEAK_GIB) >= 0.005}
    assert not disagreeing, (
        f"these statements of the measured peak disagree with the measurement of "
        f"{MEASURED_PEAK_GIB} GiB (pod, 2026-10-28, 30s cgroup-anon sampling through "
        f"pool + reconcile): {disagreeing}.\n"
        f"  Phase 2 has no preflight script, so these comments are the only thing that "
        f"stops a run being started on a box that cannot hold it — after the member "
        f"training is already paid for.\n"
        f"  Three figures for this run have already been published and withdrawn; two "
        f"of them were LOWER than the truth, which is the direction that costs money."
    )


def test_the_peak_is_not_quietly_replaced_by_the_cheaper_frame_floor() -> None:
    """The specific regression: the floor is 14.3 GiB smaller and reads like a limit.

    It is not one. `_forecast_ensemble` holds the member frames and the pooled frames
    live at the same time — no `del`, no `gc.collect` — so the frame arithmetic is a
    lower bound and the peak is the only safe sizing input.
    """
    src = _config_source()
    figures = _gib_figures(src)
    assert any(abs(f - FRAME_FLOOR_GIB) < 0.005 for f in figures), (
        f"the config no longer states the {FRAME_FLOOR_GIB} GiB frame floor, so the "
        f"peak stands with nothing to contrast it against and reads as the arithmetic "
        f"result rather than as a measurement that exceeds it."
    )
    assert MEASURED_PEAK_GIB > FRAME_FLOOR_GIB, "the test's own operands are inconsistent"
    for where, value in _peak_claims(src).items():
        assert value > FRAME_FLOOR_GIB, (
            f"the {where} gives the peak as {value} GiB, at or below the "
            f"{FRAME_FLOOR_GIB} GiB frame floor. A peak cannot be under its own "
            f"arithmetic lower bound — that is how the 27.8 GiB undersample was caught."
        )


def test_the_config_refuses_the_fifty_gigabyte_rule_by_name() -> None:
    """`RAM >= 50 GB` is the repo's general ensemble rule and it does not hold here.

    Named explicitly because an operator following the standing rule, in good faith,
    rents a box that cannot run this ensemble. The config has to contradict the rule
    out loud, not merely state a bigger number somewhere else in the file.
    """
    src = _config_source()
    assert re.search(r"RAM\s*>=\s*50\s*GB", src), (
        "the config no longer quotes the `RAM >= 50 GB` rule it exists to contradict. "
        "Stating 68.2 GiB elsewhere does not help an operator who is following the rule."
    )
    assert re.search(r"NOT ENOUGH|not enough", src), (
        "the config quotes the 50 GB rule without saying it is insufficient, which "
        "reads as endorsement."
    )


def test_the_three_superseded_figures_stay_recorded_as_superseded() -> None:
    """19.8, 27.8 and the 1.9x ratio were each wrong and each briefly believed.

    Kept in the file deliberately: a number that was published and then corrected will
    be met again in an old log or an old message, and the cheapest way to stop it being
    re-adopted is for the config to say why it is wrong. ADR-000's reasoning — superseded,
    not erased.
    """
    src = _config_source()
    for figure in ("19.8", "27.8"):
        assert figure in src, (
            f"the superseded figure {figure} GiB is no longer recorded as superseded. It "
            f"appears in the pod logs and in earlier messages; without the correction "
            f"here, the next reader meets it with nothing contradicting it."
        )


def test_the_cm_twin_does_not_present_the_pgm_peak_as_its_own() -> None:
    """`chrome_crate` is country-month. Its config must not carry chrome_box's figures.

    ⛔ The defect, found by an independent /falsify on 2026-10-30 and live for ten days:
    `chrome_crate/configs/config_hyperparameters.py` was copied from its pgm twin and the
    grid arithmetic was never replaced. It reasoned about "64,742 cells x 36 steps =
    2,330,712 rows", announced "PEAK IS ~53.8 GiB ... RAM >= 50 GB IS NOT ENOUGH", and
    concluded "87 GiB -- this is the class to rent" — for a workload measured at 96 MB of
    pool, ~2.0 MB peak RSS per member, seven minutes, on a box it never strained.

    Wrong by roughly 570x, in the expensive direction: an operator sizing phase 1 from it
    over-rents by two machine classes. This is the file-level version of the error #648
    was filed on — a pgm figure used as a cm floor — arriving by copy-paste between twins
    rather than by citation.

    The guard asserts the SHAPE of the claim, not the prose: a cm config may MENTION the
    pgm figures (it now does, to say they are not its own and must not be copied back),
    but it may not present grid-cell arithmetic as its own sizing input.
    """
    assert CRATE_CONFIG.exists(), (
        f"{CRATE_CONFIG.relative_to(REPO)} is gone; this guard cannot see its subject and "
        f"refuses rather than passing (ADR-025)."
    )
    src = CRATE_CONFIG.read_text(encoding="utf-8")

    #: The pgm row count. Allowed only where the surrounding prose disowns it.
    #:
    #: A WINDOW rather than the line itself, deliberately: prose wraps, so the number and
    #: the disclaimer that retires it routinely land on different lines. The first version
    #: of this guard checked the line alone and fired on the correction's own quotation of
    #: the old text — a guard strict enough to forbid explaining the defect it prevents.
    DISCLAIMERS = ("until", "used to", "chrome_box", "wrong", "copied", "not its own", "pgm")
    lines = src.splitlines()
    for pgm_marker in ("64,742", "2,330,712"):
        for i, line in enumerate(lines):
            if pgm_marker not in line:
                continue
            window = " ".join(lines[max(0, i - 2):i + 3]).lower()
            disowned = any(w in window for w in DISCLAIMERS)
            assert disowned, (
                f"chrome_crate's config states {pgm_marker!r} in a line that does not "
                f"disown it:\n    {line.strip()}\n"
                f"That is pgm global-land arithmetic. chrome_crate is cm — 191 countries x "
                f"36 steps, measured at 96 MB of pool and 7 minutes. Presenting the grid "
                f"figure as this ensemble's own is how an operator over-rents by two "
                f"machine classes."
            )

    assert "country-month" in src or "191 countries" in src, (
        "chrome_crate's config no longer says anywhere that it is country-month. That is "
        "the one fact which makes every number in it readable, and its absence is what let "
        "the pgm arithmetic sit here unchallenged for ten days."
    )

    assert "96 MB" in src or "96M" in src, (
        "the measured chrome_crate pool (96 MB, pod 2026-10-28, recorded in "
        "tools/podrun/pod_run_crafd_chain.sh:186) is no longer stated. Without it the file "
        "has no measurement of its own to contrast the inherited pgm figures against — "
        "which is exactly the state the defect was found in."
    )
