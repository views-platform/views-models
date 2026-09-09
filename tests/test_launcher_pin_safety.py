"""An armed delivery must not run on a views-postprocessing build with a known refusal gap.

**The gap.** Until views-postprocessing#222 the upload verification in both partner
managers read `if success is False: raise`. That **fails open**: a result that was `None`,
lacked the attribute, or carried a non-bool sailed through as though the upload had
worked. The consequence is an orphan — a file in the bucket with no metadata document —
and the consumer APIs select on metadata, so the partner sees nothing while the run
reports success. Their register calls it C-79.

**Where the fix is — RESOLVED 2026-08-13.** It used to be that C-79 lived only on
views-postprocessing's `development` (`2eb29f1`) while `main` (`3286eab`) was the newest
state carrying the crafd package, so no merged pin had both and views-models#364 blocked
two repositories. That ended when views-postprocessing tagged **`1.1.0`** (`1e21d723`,
2026-08-13 06:33): it carries the crafd package *and* `if success is not True:`.

`un_fao` moved to that tag the same morning, during the delivery that refilled the wiped
FAO buckets — and the move was not cosmetic. Verified in the installed prefix before the
run: `unfao/managers/unfao.py:71` reads `success is not True`, and
`delivery/provenance.py:57` carries `DESCRIPTION_MAX = 255`, the bound whose absence
produced the real orphan logged at 19:41 on 2026-07-27. The delivery then landed 111 files
with **111 metadata documents and zero orphans**, confirmed by reading the bucket directly.

The strict xfail below did exactly what its author intended: it XPASSed the moment the pin
moved, failed the suite, and forced its own removal. That is the mechanism, not a nuisance.

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
    # Kept after `main` gained C-79 (it resolves to 1e21d723 = tag 1.1.0 since 2026-08-13),
    # because the defect being named here is no longer the contents — it is the mutability.
    # A branch pin cannot be verified: whatever you check is not necessarily what installs.
    "main": (
        "is a BRANCH, so the build is whatever it happens to point at when the launcher "
        "runs — unverifiable by construction. It moved on 2026-08-13 at 06:33, hours "
        "before a live delivery. (It now happens to carry C-79; that is not the point.)"
    ),
    # Added 2026-08-24. 1.1.0 fixed the UPLOAD gap (C-79) and this file was written for
    # that one — which is exactly why it could not see the next one. A deficiency map
    # that learns only the defect it was born with is a guard that ages into decoration.
    "1.1.0": (
        "lacks views-postprocessing#268: the store port's `download` chains `.get()` onto "
        "an unvalidated result, so a store result whose `data` is present-and-null raises "
        "AttributeError three frames away inside a dict comprehension over pinned shard "
        "ids — naming neither the file_id nor the fact that a download failed. It killed "
        "the first un_crafd delivery on 2026-08-13 and is byte-identical on the un_fao "
        "leg, where it has simply not fired yet"
    ),
}

_PIN = re.compile(r'^VIEWS_POSTPROCESSING_PIN="([^"]+)"', re.M)
_ENV = re.compile(r'^POSTPROCESSOR_ENV_NAME="([^"]+)"', re.M)

LAUNCHERS = sorted(p.name for p in POSTPROCESSORS.iterdir() if (p / "run.sh").exists())


def _pin(consumer: str) -> str:
    match = _PIN.search((POSTPROCESSORS / consumer / "run.sh").read_text(encoding="utf-8"))
    assert match, (
        f"{consumer}/run.sh declares no VIEWS_POSTPROCESSING_PIN. Every launcher must "
        f"name the build it installs (ADR-022)."
    )
    return match.group(1)


def _env(consumer: str) -> str:
    match = _ENV.search((POSTPROCESSORS / consumer / "run.sh").read_text(encoding="utf-8"))
    assert match, f"{consumer}/run.sh declares no POSTPROCESSOR_ENV_NAME (ADR-022)."
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
    assert not _is_armed(consumer), (
        f"{consumer} is ARMED while pinned to a build that {why}.\n"
        f"  Move VIEWS_POSTPROCESSING_PIN in postprocessors/{consumer}/run.sh to a build "
        f"without that gap before arming, or set intent back to paused(...) in "
        f"deliveries/{consumer}.py.\n"
        f"  Tracked as views-models#364."
    )


def test_no_armed_launcher_is_pinned_to_a_deficient_build():
    """The state we actually want — and, since 2026-08-13, the state we are in.

    Carried a `strict=True` xfail from 2026-08-11 to 2026-08-13 because it was honestly
    false: `un_fao` was armed on `@main`, which lacked C-79. The marker was removed when
    the tag existed and the pin moved. Do not reintroduce it — if this fails, an armed
    delivery is on a build that can ship orphans, and the fix is the pin, not the marker.
    """
    offenders = {
        c: why
        for c in LAUNCHERS
        if (why := _deficiency(_pin(c))) is not None and _is_armed(c)
    }
    assert not offenders, "\n".join(f"  {c}: {why}" for c, why in offenders.items())


def test_no_launcher_can_downgrade_a_shared_prefix_under_an_armed_one():
    """Per-launcher arming is not enough: launchers SHARE a conda prefix.

    The gap this closes, found by review of PR #391. Both postprocessors declare
    `POSTPROCESSOR_ENV_NAME="views-postprocessing"` and pip-install into it. So a
    *disarmed* launcher on a deficient pin is not harmless — running it (the
    views-crafdapi D4 dry run, say) DOWNGRADES the prefix that the *armed* FAO delivery
    then uses. `tools/launcher/postprocessor.sh` has no `set -e` and no `|| return 1` on
    the pip line, so a later reinstall can fail silently, and the #294 capability
    assertion still passes on the stale build because it also carries `contract/wire`.
    That is a live path back to C-135 on a UN-facing delivery.

    The sibling test asks "is THIS launcher armed on a bad pin?" and answers no for a
    paused consumer — correctly, and uselessly, because the danger is to its neighbour.
    This asks the question that matters: **if anyone sharing this prefix is armed, every
    launcher writing to it must be on a sound pin.**
    """
    by_prefix = {}
    for consumer in LAUNCHERS:
        by_prefix.setdefault(_env(consumer), []).append(consumer)

    offenders = []
    for prefix, consumers in sorted(by_prefix.items()):
        armed = [c for c in consumers if _is_armed(c)]
        if not armed:
            continue
        for consumer in consumers:
            why = _deficiency(_pin(consumer))
            if why is not None:
                offenders.append((prefix, consumer, armed, why))

    assert not offenders, "\n".join(
        f"  prefix {prefix!r}: {consumer} is pinned to a build that {why}\n"
        f"    — and {', '.join(armed)} is ARMED on the same prefix, so running "
        f"{consumer} downgrades the build {', '.join(armed)} delivers with.\n"
        f"    Move VIEWS_POSTPROCESSING_PIN in postprocessors/{consumer}/run.sh, or "
        f"give it its own prefix."
        for prefix, consumer, armed, why in offenders
    )


def test_the_launcher_verifies_the_pin_it_installed():
    """Declaring a pin is not the same as installing it (#385).

    views-postprocessing declares a STATIC version, so pip treats the requirement as
    satisfied whenever any build of that version is present and skips the rebuild. Moving
    the pin to a different COMMIT of the same version is a silent no-op on any machine
    that has run the launcher before — found during the first CRAF'd delivery
    (views-crafdapi#44).

    Pinning a tag mostly dodges it, and both launchers now pin tags. That is a property
    of today's pins, not of the mechanism: the next raw-commit pin gets the old behaviour
    with no warning. So the body must CHECK, and the check must abort rather than warn —
    the whole failure mode is a run that proceeds on the wrong build believing it is
    right.
    """
    body = (REPO_ROOT / "tools" / "launcher" / "postprocessor.sh").read_text(encoding="utf-8")
    assert "direct_url.json" in body, (
        "the shared launcher body does not read direct_url.json, so it cannot know which "
        "build pip actually installed (#385)."
    )
    assert "installed_ref" in body and "VIEWS_POSTPROCESSING_PIN" in body, (
        "no comparison of the installed ref against the declared pin (#385)."
    )
    # It must ABORT on a mismatch. A warning here would be worse than nothing: it reads as
    # a check while still running the wrong build against a partner bucket.
    after = body.split("installed_ref", 1)[1]
    assert "return 1" in after, (
        "the pin mismatch does not abort. A mismatch means the build about to deliver is "
        "not the one declared — that must stop the run, not warn (#385, ADR-020)."
    )
