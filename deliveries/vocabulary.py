"""The small language a delivery file is written in (ADR-019).

A delivery file names a consumer by its filename and declares two blocks:

    DELIVERY = Delivery(...)   # DECIDES  — change a line, something different happens
    REQUIRE  = Require(...)    # REFUSES  — change a line, a different set is rejected

Everything here is either a type or a constructor for those two blocks. It is one
concept — the delivery language — so it lives in one module.

Deliberately minimal. There is no `__str__`, no `is_live`, no convenience predicate,
because nothing calls one yet — the checks that will format these values into error
messages are views-models#344 and #345, and they can add what they actually use. WET
before DRY applies to a vocabulary as much as to a framework.

What this module does *not* do: check that a source exists, that a level claim matches
the source's own config, or that a reconciliation graph is connected. Those are
cross-file coherence rules and belong with the checks (ADR-019 §4, views-models#344).
What is enforced here is only what a single value can be wrong about on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

__all__ = [
    "Delivery", "Require", "Source", "Intent", "Months",
    "pgm", "cm", "live", "paused", "months",
    "monthly", "prod",
]


# ── Sentinels ──────────────────────────────────────────────────────────────
# Named objects rather than bare strings, so a typo is a NameError that points at the
# file, instead of a string that silently means nothing (ADR-020).

@dataclass(frozen=True)
class _Word:
    name: str


#: The only frequency today. ADR-019 §3: the key exists so a second cadence is one new
#: word rather than a schema change touching every existing file.
monthly = _Word("monthly")

#: The only consumer tier today. ADR-019 §3: `prod` means every source must be
#: `graduate`. A second value is blocked on ADR-017 §12's shadow-destination question.
prod = _Word("prod")


# ── What is sent ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Source:
    """A forecast source, with its level *claimed* rather than set.

    ADR-019 §3: `pgm("x")` does not make x pgm. The level already lives on the source
    (`"level": "pgm"`); writing it here states what you believe, and the system refuses
    if the source disagrees. That correspondence check is #344's, not this module's.
    """

    name: str
    level: str


def pgm(name: str) -> Source:
    """Claim that `name` is a grid-cell (PRIO-GRID month) source."""
    return Source(name=name, level="pgm")


def cm(name: str) -> Source:
    """Claim that `name` is a country-month source."""
    return Source(name=name, level="cm")


# ── Whether it ships ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class Intent:
    """Declared intent. Status is derived elsewhere and never typed (ADR-017 §4e)."""

    state: str
    since: date
    reason: str | None = None


def live(since: date) -> Intent:
    """Armed: a runner picks this delivery up at its frequency.

    `since` is required. ADR-020 §4 calls the live-but-never-run case "the hole in the
    floor" — nothing fails, so nothing is raised. A declared start date does not close
    it, but it makes the silence *measurable*: `since` minus the last observed delivery
    is how long a declared-live edge has produced nothing (register C-126, #320).
    """
    if not isinstance(since, date):
        raise TypeError(
            f"live(since=...) needs a datetime.date, got {type(since).__name__}.\n"
            f"  Write: live(since=date(2026, 8, 4))"
        )
    return Intent(state="live", since=since)


def paused(reason: str, since: date) -> Intent:
    """Not armed: the runner skips this delivery.

    Both arguments are required. ADR-019 §3: you cannot switch a delivery off silently,
    because the disease being treated is two halves quietly waiting with nobody able to
    see it. A pause carrying a six-month-old date is a visible fact; an absence is not.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(
            "paused(...) needs a reason explaining why, in words the next person can act on.\n"
            '  Write: paused("waiting on the OCHA bucket — ask Simon", since=date(2026, 8, 4))'
        )
    if not isinstance(since, date):
        raise TypeError(
            f"paused(..., since=...) needs a datetime.date, got {type(since).__name__}.\n"
            f"  Without a date, nobody can tell a new pause from a forgotten one."
        )
    return Intent(state="paused", since=since, reason=reason)


# ── How old is too old ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class Months:
    count: int


def months(count: int) -> Months:
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError(
            f"months(...) needs a positive whole number of months, got {count!r}."
        )
    return Months(count=count)


# ── The two blocks ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Delivery:
    """What happens. Change a line here and something different is produced."""

    send: tuple[Source, ...]
    frequency: _Word
    tier: _Word
    intent: Intent

    def __post_init__(self) -> None:
        if isinstance(self.send, Source):
            raise TypeError(
                "send must be a list, even with one source: send=[pgm('x')].\n"
                "  ADR-017 §3: a consumer may need a grid-cell forecast and the "
                "country-level forecast it was reconciled against."
            )
        if not self.send:
            raise ValueError("send must name at least one source.")
        object.__setattr__(self, "send", tuple(self.send))


@dataclass(frozen=True)
class Require:
    """What is refused. Removing an *optional* line here widens what is allowed
    through; it never changes what is produced.

    `max_age` is the one exception, and it is mandatory for a `live()` delivery
    (ADR-019 §4). Removing it does not widen anything — it makes the file invalid.
    That asymmetry is deliberate: a missing freshness bound is the failure that
    already happened, for five months (#320, register C-121, C-128).
    """

    targets: tuple[str, ...] = ()
    reconciled: bool | None = None
    coverage: str | None = None
    max_age: Months | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))
