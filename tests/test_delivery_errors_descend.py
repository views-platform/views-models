"""Errors from a delivery file must send the reader one level down (ADR-020).

Two kinds of test live here, and the second is the point of the story.

**Per-failure tests** assert that each failure class names the file to open next.
**The meta-test** enumerates every `raise` site in `deliveries/` *statically* and asserts
the same thing — including sites no test happens to provoke.

The static form matters. ADR-020 §3 predicts the failure mode: *"the staircase rots in
its second month — someone refactors a check, the message becomes `KeyError: 'level'`,
and nothing fails."* It did not take a month. While writing #344, an audit of its ten
raise sites found one that named no file, because `_check_reconciliation` never received
the `consumer` argument and so *could not* name the file even in principle. Nine of ten
were right; nothing would have failed. A dynamic meta-test that only inspects errors it
manages to trigger would have missed it, because no fixture reached that path.

Assertions are on **substance, not wording**: that a path appears, not that a sentence
matches. Brittle string equality would make every message edit a test failure, and would
teach the next person to delete the test rather than fix the message.
"""

import ast
import re
from datetime import date
from pathlib import Path

import pytest

from deliveries import coherence
from deliveries.coherence import CoherenceError, check
from deliveries.vocabulary import Delivery, Require, cm, live, monthly, months, pgm, prod

pytestmark = pytest.mark.red

REPO_ROOT = Path(__file__).resolve().parents[1]
DELIVERIES_DIR = REPO_ROOT / "deliveries"

#: A message "names the next file" if it contains something a reader can open.
NAMES_A_FILE = re.compile(r"[\w./-]+\.py\b|\b(?:models|ensembles|deliveries)/")


def _delivery(send, **require_kwargs):
    require_kwargs.setdefault("max_age", months(2))
    return (
        Delivery(
            send=send,
            frequency=monthly,
            tier=prod,
            intent=live(since=date(2026, 8, 4)),
        ),
        Require(**require_kwargs),
    )


# ── The staircase, one step at a time (ADR-020 §2) ─────────────────────────


class TestEachFailureNamesTheNextFile:
    def test_wrong_level_claim_points_at_the_sources_config(self):
        delivery, require = _delivery([cm("rusty_bucket")])
        with pytest.raises(CoherenceError) as exc:
            check(delivery, require, consumer="un_fao")
        assert "ensembles/rusty_bucket/configs/config_meta.py" in str(exc.value)

    def test_unknown_source_points_at_where_sources_live(self):
        delivery, require = _delivery([pgm("no_such_ensemble")])
        with pytest.raises(CoherenceError) as exc:
            check(delivery, require, consumer="un_fao")
        message = str(exc.value)
        assert "models/" in message and "ensembles/" in message

    def test_unknown_source_suggests_the_closest_real_name(self):
        """ADR-020 §2: '...and the closest name to what was typed'."""
        delivery, require = _delivery([pgm("rusty_buckt")])
        with pytest.raises(CoherenceError) as exc:
            check(delivery, require, consumer="un_fao")
        assert "rusty_bucket" in str(exc.value)

    def test_disconnected_reconciliation_points_at_the_partner_declaration(self):
        delivery, require = _delivery(
            [pgm("skinny_love"), cm("rude_boy")], reconciled=True
        )
        with pytest.raises(CoherenceError) as exc:
            check(delivery, require, consumer="un_fao")
        message = str(exc.value)
        assert "config_meta.py" in message
        assert "reconcile_with" in message

    def test_missing_freshness_points_at_the_delivery_file(self):
        delivery, require = _delivery([pgm("rusty_bucket")], max_age=None)
        with pytest.raises(CoherenceError) as exc:
            check(delivery, require, consumer="un_fao")
        assert "deliveries/un_fao.py" in str(exc.value)

    def test_no_error_ends_in_a_bare_exception_type(self):
        """A reader sees the last line of a traceback. It must not be a KeyError."""
        delivery, require = _delivery([pgm("no_such_ensemble")])
        with pytest.raises(CoherenceError) as exc:
            check(delivery, require, consumer="un_fao")
        assert len(str(exc.value)) > 40, "an error this short cannot be teaching anything"


# ── The meta-test: what stops this rotting ─────────────────────────────────


def _raise_sites(path: Path) -> list[tuple[int, str]]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and node.exc is not None:
            segment = ast.get_source_segment(source, node) or ""
            sites.append((node.lineno, segment))
    return sites


class TestMetaEveryRaiseSiteDescends:
    """Enumerate raise sites statically, so a path no test provokes is still checked."""

    def test_there_are_raise_sites_to_check(self):
        """Guard against the meta-test silently checking nothing.

        If `deliveries/` is restructured and this finds zero sites, the suite would
        go green while enforcing nothing — the same shape as the C-113 defect.
        """
        total = sum(len(_raise_sites(p)) for p in DELIVERIES_DIR.glob("*.py"))
        assert total >= 5, (
            f"only {total} raise sites found across deliveries/*.py — this meta-test "
            f"is probably no longer looking where the checks live."
        )

    def test_every_check_raise_names_a_file(self):
        """`coherence.py` compares files. The reader is looking at the delivery file
        and the problem is in a *different* one, so the message must name it."""
        offenders = [
            f"coherence.py:{lineno}"
            for lineno, segment in _raise_sites(DELIVERIES_DIR / "coherence.py")
            if not NAMES_A_FILE.search(segment)
        ]
        assert not offenders, (
            f"these raise sites name no file to open: {offenders}\n"
            f"  ADR-020 §2: an error must send the reader exactly one level down, "
            f"naming the next file.\n"
            f"  If the site genuinely cannot name one — because the answer is outside "
            f"this repository — use locked_door() instead (ADR-020 §5)."
        )

    def test_every_vocabulary_raise_shows_what_to_write(self):
        """`vocabulary.py` raises while the delivery file is being *constructed*, so
        Python's traceback already names that file and the exact line. Demanding a
        filename here would mean guessing one — the constructor cannot know which
        delivery called it.

        The obligation is therefore different, not absent: show the corrected form.
        Verified by hand: a failure in `live()` from a delivery file produces a
        traceback whose frames name that file. The two rules together are what
        ADR-020 §2 means by "exactly one level down" — sometimes down is the line
        you are already on.
        """
        offenders = [
            f"vocabulary.py:{lineno}"
            for lineno, segment in _raise_sites(DELIVERIES_DIR / "vocabulary.py")
            if "Write:" not in segment and "send=[" not in segment
        ]
        assert not offenders, (
            f"these vocabulary errors say what is wrong but not what to write: "
            f"{offenders}\n"
            f"  Add a corrected example, e.g. 'Write: months(2)'.\n"
            f"  The reader is a research assistant who cannot infer the right form "
            f"from a type name (ADR-020 §1)."
        )

    def test_no_module_in_deliveries_escapes_both_rules(self):
        """Guards the split above: a new module in deliveries/ must be assigned to
        one rule or the other, not silently checked by neither."""
        covered = {"coherence.py", "vocabulary.py", "__init__.py", "un_fao.py"}
        present = {p.name for p in DELIVERIES_DIR.glob("*.py")}
        unassigned = {
            name for name in present - covered
            if _raise_sites(DELIVERIES_DIR / name)
        }
        assert not unassigned, (
            f"{sorted(unassigned)} raise errors but are covered by neither rule.\n"
            f"  Open tests/test_delivery_errors_descend.py and decide which applies: "
            f"cross-file checks must name a file; construction errors must show what "
            f"to write."
        )


class TestMetaKnownLimit:
    """The static scan has one blind spot. Stating it beats implying it is complete."""

    def test_helpers_that_build_messages_elsewhere_are_not_covered(self):
        """A check that raises via a helper hides its message from the scan.

        `deliveries/coherence.py` has one such helper, `_require_source`, and it is
        covered by `TestEachFailureNamesTheNextFile` above. This test pins that the
        helper still produces a descending message, since the meta-test cannot.
        """
        with pytest.raises(CoherenceError) as exc:
            coherence._require_source("definitely_not_a_source")
        assert NAMES_A_FILE.search(str(exc.value)), (
            "_require_source raises from a helper, so the static meta-test cannot see "
            "its message. It must be checked here instead."
        )


# ── Locked doors (ADR-020 §5) ──────────────────────────────────────────────


class TestLockedDoor:
    """Where the stairs end, the error names a person and confirms the rest is fine."""

    def test_message_names_a_person_and_a_ready_made_request(self):
        message = coherence.locked_door(
            what="un_ocha is not a registered consumer",
            why="Registering one needs a bucket address from the platform coordinate "
                "registry, which is in another repository you are not expected to edit",
            request='Register consumer un_ocha (bucket + API)',
        )
        assert "Simon" in message
        assert "open an issue" in message

    def test_message_confirms_the_rest_of_the_work_is_fine(self):
        """ADR-020 §5: 'that last line is the difference between a handoff and a
        dead end.' Without it, this is where people give up and ask someone else."""
        message = coherence.locked_door(what="x", why="y", request="z")
        assert "Everything else in this file is fine" in message

    def test_message_does_not_end_in_a_task_the_reader_cannot_perform(self):
        """ADR-020 §1: the reader cannot publish a package or edit another repo."""
        message = coherence.locked_door(what="x", why="y", request="z")
        last = [line for line in message.strip().splitlines() if line.strip()][-1]
        assert "fine" in last or "blocking" in last
