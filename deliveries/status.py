"""Derived delivery status — "in production" is worked out, never typed (ADR-017 §4e).

    A source is *in production* ⟺ its maturity is `graduate` **and** a delivery ships
    it (directly, or via a composite that contains it) to a **production-tier** consumer.

Nobody writes "deployed" anywhere. The answer is computed from the source's own maturity
and the delivery edges in `deliveries/`, so there is no field that can lie.

**What this module does not do.** It reads declarations. It observes nothing: no bucket,
no run, no artifact. A delivery declared `live()` that no runner ever executes is
invisible here, because nothing failed — ADR-020 §4's "hole in the floor", register
C-126, and the failure that actually happened (#320). The observation side is
`python -m tools.liveness`, and `report()` says so in its own output rather than leaving
a reader to assume it checked.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from deliveries.coherence import CoherenceError, maturity_of, require_source, source_config

DELIVERIES_DIR = Path(__file__).resolve().parent
REPO_ROOT = DELIVERIES_DIR.parent

#: Modules in deliveries/ that are machinery, not declarations.
_NOT_A_DELIVERY = {"__init__", "vocabulary", "coherence", "status"}


def delivery_files() -> list[Path]:
    """Every `deliveries/<consumer>.py`. The filename is the consumer (ADR-019 §1)."""
    return sorted(
        path for path in DELIVERIES_DIR.glob("*.py")
        if path.stem not in _NOT_A_DELIVERY
    )


def load_delivery(path: Path):
    spec = importlib.util.spec_from_file_location(f"_delivery_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _members_of(source: str) -> list[str]:
    return source_config(source, "modelset").get("models", [])


def delivered_sources() -> dict[str, list[str]]:
    """Every source reachable from a delivery, mapped to the consumers reached.

    Walks composites: ADR-017 §4a says a model inside a delivered ensemble is in
    production *transitively*, so naming the ensemble is not enough — the members
    have to be followed.
    """
    reached: dict[str, list[str]] = {}

    def visit(name: str, consumer: str, seen: frozenset[str]) -> None:
        if name in seen:
            raise CoherenceError(
                f"'{name}' is a member of itself, directly or through "
                f"{' -> '.join(sorted(seen))}.\n"
                f"  Open ensembles/{name}/configs/config_modelset.py and break the cycle."
            )
        reached.setdefault(name, [])
        if consumer not in reached[name]:
            reached[name].append(consumer)
        for member in _members_of(name):
            visit(member, consumer, seen | {name})

    for path in delivery_files():
        module = load_delivery(path)
        for source in module.DELIVERY.send:
            visit(source.name, path.stem, frozenset())
    return reached


def declared_max_age_days(consumer: str = "un_fao") -> int:
    """The freshness bound this consumer's delivery declares, in days.

    The *only* freshness threshold for a delivery. Instruments that measure
    staleness read it from here rather than carrying their own number — two
    thresholds in two files is what ADR-019 §8 rejects, and it is what
    `tools/liveness/unfao_delivery.py` used to do with `DELIVERING_WITHIN_DAYS = 45`
    while this file declared two months (register C-121).

    Raises rather than defaulting. A fallback bound would re-create the same defect
    with one of the two numbers invisible.
    """
    path = DELIVERIES_DIR / f"{consumer}.py"
    if not path.exists():
        raise FileNotFoundError(
            f"no delivery declaration for '{consumer}'.\n"
            f"  Expected: deliveries/{consumer}.py\n"
            f"  The freshness bound is declared there; this will not invent one."
        )
    require = load_delivery(path).REQUIRE
    if require.max_age is None:
        raise ValueError(
            f"deliveries/{consumer}.py declares no max_age, so there is no bound to "
            f"measure staleness against.\n"
            f"  Add max_age=months(n) to REQUIRE.\n"
            f"  A live delivery must declare one (ADR-019 §4)."
        )
    # Months are a declaration unit, not a calendar computation: the bound exists to
    # answer "has a monthly delivery been missed?", where 30-day months are exact
    # enough and, unlike calendar arithmetic, do not vary by when you ask.
    return require.max_age.count * 30


def consumers_for(source: str) -> list[str]:
    """Which consumers this source reaches, directly or as a member."""
    return delivered_sources().get(source, [])


def in_production(source: str) -> bool:
    """ADR-017 §4e's conjunction. Computed on demand; never stored.

    The second conjunct is written for the general case. `tier` has exactly one value
    today (ADR-019 §3), so "to a production-tier consumer" is currently equivalent to
    "at all" — do not read it as a check that is discriminating yet (register C-131).
    """
    require_source(source)
    if maturity_of(source) != "graduate":
        return False
    # Must be reached by a delivery that is ITSELF production-tier. Checking "some
    # prod delivery exists" and "this source is delivered somewhere" separately
    # would conflate two different deliveries — currently indistinguishable, because
    # `tier` has one value, and wrong the moment it has two (register C-131).
    return any(
        consumer in _prod_tier_consumers()
        for consumer in consumers_for(source)
    )


def _prod_tier_consumers() -> set[str]:
    return {
        path.stem for path in delivery_files()
        if load_delivery(path).DELIVERY.tier.name == "prod"
    }


def report() -> str:
    """A read-only summary of what the repository *declares* about its deliveries."""
    lines = [
        "Declared delivery state",
        "=" * 60,
        "",
        "Everything below is DECLARED, not observed. No bucket was read and no run",
        "was checked. A delivery declared live that no runner executes looks",
        "exactly like one that shipped this morning (ADR-020 §4, register C-126).",
        "For what actually happened, run:  python -m tools.liveness",
        "",
    ]

    paths = delivery_files()
    if not paths:
        lines.append("No delivery files. Expected deliveries/<consumer>.py (ADR-019 §1).")
        return "\n".join(lines)

    for path in paths:
        module = load_delivery(path)
        delivery, require = module.DELIVERY, module.REQUIRE
        intent = delivery.intent
        state = intent.state
        if intent.reason:
            state += f" — {intent.reason}"

        lines += [
            f"{path.stem}",
            f"  declared in   deliveries/{path.name}",
            f"  frequency     {delivery.frequency.name}",
            f"  tier          {delivery.tier.name}",
            f"  intent        {state}  (since {intent.since})",
            "  sends",
        ]
        for source in delivery.send:
            maturity = maturity_of(source.name)
            produced = "in production" if in_production(source.name) else "not in production"
            lines.append(
                f"    {source.level}({source.name})  maturity={maturity}  → {produced}"
            )
            members = _members_of(source.name)
            if members:
                lines.append(f"      via {len(members)} members, transitively")

        if require.max_age:
            lines.append(f"  max_age       {require.max_age.count} months")
        else:
            lines.append("  max_age       none declared")
        if require.targets:
            lines.append(f"  targets       {', '.join(require.targets)}  (not checked here)")
        if require.coverage:
            lines.append(f"  coverage      {require.coverage}  (not checked here)")
        lines.append("")

    lines += [
        "-" * 60,
        "'in production' is derived from maturity + a delivery edge (ADR-017 §4e).",
        "It is not stored anywhere, so there is no field that can be wrong.",
        "",
        "targets and coverage are declared but not verified here — both are answered",
        "outside this repository (ADR-020 §4).",
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(report())
