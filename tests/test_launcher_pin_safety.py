"""An armed delivery must not run on a views-postprocessing build with a known refusal gap.

**The gap.** Until views-postprocessing#222 the upload verification in both partner
managers read `if success is False: raise`. That **fails open**: a result that was `None`,
lacked the attribute, or carried a non-bool sailed through as though the upload had
worked. The consequence is an orphan — a file in the bucket with no metadata document —
and the consumer APIs select on metadata, so the partner sees nothing while the run
reports success. Their register calls it C-79.

**Where the fix is.** On views-postprocessing's `development` (`2eb29f1`), not on `main`
(`3286eab`). `main` is 33 commits behind and is the newest state that carries the crafd
package at all — the only tag, `1.0.0`, predates it. So today there is **no** pin that has
both crafd and C-79 on a merged branch, which is why views-models#364 (cut a tag) now
blocks two repositories.

**What this file asserts.** Not "the pin is current" — that would be a moving target and a
nag. Only the narrow thing that matters: **a delivery that is armed must not be pinned to a
build whose upload check fails open.** A disarmed launcher makes zero store calls, so the
gap cannot reach it; the danger is exactly the transition.
"""

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.beige]

REPO_ROOT = Path(__file__).resolve().parents[1]
POSTPROCESSORS = REPO_ROOT / "postprocessors"

#: views-postprocessing builds known to be missing a delivery refusal, and why. Keyed by
#: the prefix a pin would carry. Entries leave this map when the fix reaches a merged
#: branch and the pins move — not before.
DEFICIENT_PINS = {
    "3286eab": (
        "lacks C-79 (views-postprocessing#222): the upload check reads "
        "`if success is False`, so a None or non-bool result is treated as a successful "
        "upload and leaves an orphan file with no metadata document"
    ),
    "main": (
        "resolves to views-postprocessing main (3286eab today), which lacks C-79 — and "
        "being a branch it can move under you without the pin changing"
    ),
}

_PIN = re.compile(r'^VIEWS_POSTPROCESSING_PIN="([^"]+)"', re.M)

LAUNCHERS = sorted(p.name for p in POSTPROCESSORS.iterdir() if (p / "run.sh").exists())


def _pin(consumer: str) -> str:
    match = _PIN.search((POSTPROCESSORS / consumer / "run.sh").read_text(encoding="utf-8"))
    assert match, (
        f"{consumer}/run.sh declares no VIEWS_POSTPROCESSING_PIN. Every launcher must "
        f"name the build it installs (ADR-022)."
    )
    return match.group(1)


def _deficiency(pin: str):
    for prefix, why in DEFICIENT_PINS.items():
        if pin.startswith(prefix):
            return why
    return None


def _is_armed(consumer: str) -> bool:
    from deliveries.status import upload_armed

    return upload_armed(consumer)


def test_there_is_at_least_one_launcher_to_check():
    assert LAUNCHERS, "no launchers discovered — the checks below assert nothing"


@pytest.mark.parametrize("consumer", LAUNCHERS)
def test_a_disarmed_launcher_stays_disarmed_while_its_pin_is_deficient(consumer):
    """The guard that matters: arming and the pin must move together.

    This passes for a disarmed launcher on any pin, and for an armed launcher on a good
    pin. It fails only on the combination that ships orphans — which is what someone
    flipping `intent` to `live()` without touching the pin would create.
    """
    why = _deficiency(_pin(consumer))
    if why is None:
        return
    if consumer == "un_fao":
        pytest.skip("un_fao's pre-existing state is documented by the xfail below")
    assert not _is_armed(consumer), (
        f"{consumer} is ARMED while pinned to a build that {why}.\n"
        f"  Move VIEWS_POSTPROCESSING_PIN in postprocessors/{consumer}/run.sh to a build "
        f"carrying views-postprocessing#222 before arming, or set intent back to "
        f"paused(...) in deliveries/{consumer}.py.\n"
        f"  Tracked as views-models#364."
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "un_fao is live and pinned to @main, which lacks C-79 — a real, current gap on "
        "the UN-facing delivery, not a hypothetical. It cannot be fixed here: no merged "
        "views-postprocessing branch carries both the crafd package and C-79. Flips to "
        "XPASS (and so fails, forcing this marker's removal) the day the pin moves — "
        "views-models#364."
    ),
)
def test_no_armed_launcher_is_pinned_to_a_deficient_build():
    """The state we actually want, recorded as expected-to-fail rather than omitted."""
    offenders = {
        c: why
        for c in LAUNCHERS
        if (why := _deficiency(_pin(c))) is not None and _is_armed(c)
    }
    assert not offenders, "\n".join(f"  {c}: {why}" for c, why in offenders.items())
