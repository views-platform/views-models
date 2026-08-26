"""The coherence rules a delivery file must satisfy (ADR-019 §4, ADR-017 §5).

These live beside the declaration rather than in `tools/`, because they are a contract,
not an instrument: `tools/` observes, this refuses.

**Everything here is answerable offline, inside this repository.** Two rules from
ADR-019 §4 are deliberately absent, and their absence is a decision recorded in
ADR-020 §4:

- **`targets`** — whether a target *exists* is checked against a real run's manifests,
  not a config. A gate here today would reject a *correct* delivery file, because
  `rusty_bucket` declares `lr_*_best` and emits `lr_ged_*` (register C-123). The first
  thing this repository would teach a newcomer is that its errors are wrong (C-125).

  `_check_target_coverage` is **not** that gate and does not weaken this (#428). It compares
  `REQUIRE.targets` against the `provides=` written beside it in the same file — two
  strings in one namespace, no source config consulted. A file can pass it and still
  name a target no run will ever contain.
- **`coverage`** — the cell counts defining a region live in views-postprocessing,
  beside the GAUL asset. They belong there.

**On maturity.** `maturity` does not exist yet: the `deployment_status` rename is
ADR-017 §11 Phase 2 and is cross-repo (views-pipeline-core#398). So the rules run
against today's field with ADR-017 §3's mapping applied in memory. Phase 2 then changes
what the values are *called*, not what the rules *say*.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
ENSEMBLES_DIR = REPO_ROOT / "ensembles"


class CoherenceError(Exception):
    """A delivery file is incoherent. The message names the next file to open."""


#: Who to ask when the staircase ends outside this repository (ADR-020 §1, §5).
MAINTAINER = "Simon"


def locked_door(*, what: str, why: str, request: str) -> str:
    """Compose the message for a check that cannot be answered in this repository.

    ADR-020 §5: name the person, supply the request, and confirm the rest of the work
    is fine. That last part is not politeness — it is the difference between a handoff
    and a dead end. A locked door that also tells you the rest of your work is correct
    is a good place to stop; one that does not is where people give up and ask someone
    else to do it for them, which is how delivery became undiscoverable in the first
    place (ADR-017 §2).

    The audience cannot publish a package or edit another repository (ADR-020 §1), so
    no message composed here may end in a task they cannot perform.
    """
    return (
        f"{what}.\n\n"
        f"  {why}.\n\n"
        f'  Ask {MAINTAINER}, or open an issue: "{request}".\n'
        f"  Everything else in this file is fine — this is the only thing blocking it."
    )


# ── Reading a source's own declarations ────────────────────────────────────


def _load(path: Path, unique_name: str):
    """Load a config file as a module.

    `tests/conftest.py` has a near-identical helper. This is a deliberate copy, not an
    oversight: production code importing from the test package would invert the
    dependency and make `deliveries/` unusable without pytest installed. Ten duplicated
    lines are cheaper than that coupling — WET before DRY, and the right seam to share
    across is not yet known.
    """
    spec = importlib.util.spec_from_file_location(unique_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_dir(name: str) -> Path | None:
    for base in (ENSEMBLES_DIR, MODELS_DIR):
        candidate = base / name
        if (candidate / "configs").is_dir():
            return candidate
    return None


def source_config(source: str, which: str) -> dict:
    """Load one of a source's config dicts, or {} if that config does not exist."""
    directory = _source_dir(source)
    if directory is None:
        return {}
    path = directory / "configs" / f"config_{which}.py"
    if not path.exists():
        return {}
    module = _load(path, f"_delivery_cfg_{source}_{which}")
    getter = getattr(module, f"get_{which}_config", None)
    return getter() if getter else {}


def require_source(name: str) -> Path:
    directory = _source_dir(name)
    if directory is None:
        raise CoherenceError(
            f"'{name}' is not a source in this repository.\n"
            f"  Looked in: models/{name}/configs/ and ensembles/{name}/configs/\n"
            f"  Open ensembles/ and check the spelling{_did_you_mean(name)}."
        )
    return directory


def _did_you_mean(name: str) -> str:
    import difflib

    known = [p.name for base in (ENSEMBLES_DIR, MODELS_DIR)
             for p in base.iterdir() if (p / "configs").is_dir()]
    close = difflib.get_close_matches(name, known, n=1)
    return f" — did you mean '{close[0]}'?" if close else ""


# ── Maturity (ADR-017 §3's migration mapping, applied in memory) ───────────

_MATURITY = {"shadow": "candidate", "baseline": "candidate", "deprecated": "retired"}


def maturity_of(source: str, _seen: frozenset[str] = frozenset()) -> str:
    """Today's `deployment_status`, expressed in ADR-017's maturity vocabulary."""
    if source in _seen:
        raise CoherenceError(
            f"'{source}' is a member of itself, directly or through "
            f"{' -> '.join(sorted(_seen))}.\n"
            f"  Open ensembles/{source}/configs/config_modelset.py and break the cycle.\n"
            f"  An ensemble cannot contain itself; its maturity would have no answer."
        )
    require_source(source)
    status = source_config(source, "deployment").get("deployment_status")
    if status is None:
        # Four source directories carry no config_deployment.py at all (ADR-017 §3).
        return "candidate"
    if status in _MATURITY:
        return _MATURITY[status]
    if status == "deployed":
        # ADR-017 §3: `graduate` only where R2 already holds, else `candidate`. A
        # straight rename would make the repo's one `deployed` ensemble a graduate
        # with candidate members — a violation of ADR-017's own rule on day one.
        members = source_config(source, "modelset").get("models", [])
        deeper = _seen | {source}
        if members and all(maturity_of(m, deeper) == "graduate" for m in members):
            return "graduate"
        return "candidate"
    raise CoherenceError(
        f"'{source}' declares an unknown deployment_status '{status}'.\n"
        f"  Open {_source_dir(source).relative_to(REPO_ROOT)}/configs/config_deployment.py\n"
        f"  Valid today: shadow, deployed, baseline, deprecated."
    )


# ── The rules ──────────────────────────────────────────────────────────────


def _check_resolution_and_level(delivery) -> None:
    for source in delivery.send:
        directory = require_source(source.name)
        declared = source_config(source.name, "meta").get("level")
        if declared is None:
            raise CoherenceError(
                f"{source.level}('{source.name}') cannot be checked: that source "
                f"declares no level.\n"
                f"  Open {directory.relative_to(REPO_ROOT)}/configs/config_meta.py "
                f"and add \"level\"."
            )
        if declared != source.level:
            raise CoherenceError(
                f"the delivery claims {source.level}('{source.name}') but that source "
                f"declares level '{declared}'.\n"
                f"  Open {directory.relative_to(REPO_ROOT)}/configs/config_meta.py — "
                f"one of the two is wrong.\n"
                f"  Neither is authoritative over the other (ADR-019 §4); they must agree."
            )


def _check_target_coverage(delivery, require, consumer: str) -> None:
    """Every required target is claimed by exactly one source at a level (ADR-019 §4).

    Named `target_coverage` and not `coverage`, because `Require.coverage` in the same
    file is an unrelated thing — a GAUL region name, whose cell counts live in
    views-postprocessing and are not checked here at all.

    This is the *other* reason a delivery names several sources. Reconciliation says the
    sources agree with each other about one target; coverage says that between them they
    carry the targets asked for. `un_crafd` needs three, every reconciling ensemble
    carries one, and the ensemble that carries three reconciles with nothing (#424).

    **It compares two things written in the same file, and nothing else.** It is not
    evidence that the targets exist: `ensembles/rusty_bucket/configs/config_meta.py`
    declares `lr_*_best` while both deliveries REQUIRE `lr_ged_*` — different strings,
    register C-123. Checking a target against a source config would refuse a *correct*
    delivery file, which is why that stair is deliberately absent (module docstring,
    ADR-020 §4). Nothing here changes that.

    Same target at two *different* levels is the reconciliation case (ADR-017 §3) and is
    allowed; the same target twice at one level is two answers to one question.
    """
    if len(delivery.send) < 2:
        return

    annotated = [s for s in delivery.send if s.provides is not None]
    if not annotated:
        # `provides` omitted throughout means "everything this source contains", so
        # nothing is claimed exclusively and there is nothing here to be wrong about.
        return
    if len(annotated) != len(delivery.send):
        silent = [s.name for s in delivery.send if s.provides is None]
        raise CoherenceError(
            f"deliveries/{consumer}.py annotates some sources with provides= but not "
            f"{', '.join(silent)}.\n"
            f"  Open deliveries/{consumer}.py and either give every source a provides=, "
            f"or remove them all.\n"
            f"  An un-annotated source claims every target it contains, so it overlaps "
            f"whatever the others claim and the division stops meaning anything."
        )

    claims: dict[tuple[str, str], list[str]] = {}
    for source in annotated:
        for target in source.provides:
            claims.setdefault((source.level, target), []).append(source.name)

    for (level, target), sources in sorted(claims.items()):
        if len(sources) > 1:
            raise CoherenceError(
                f"deliveries/{consumer}.py has '{target}' claimed by "
                f"{' and '.join(sources)}, both at level {level}.\n"
                f"  Open deliveries/{consumer}.py and remove '{target}' from one of "
                f"their provides=.\n"
                f"  Two sources at one level answering for one target is two answers "
                f"to one question; the consumer has no rule for choosing.\n"
                f"  (The same target at pgm *and* cm is different — that is "
                f"reconciliation, and it is allowed.)"
            )

    # `Require.targets` defaults to `()`, so a delivery that states no targets has
    # nothing to be missing — but the duplicate rule above still applies to it, because
    # two sources contradicting each other is wrong whether or not anyone asked.
    claimed = {target for _level, target in claims}
    unclaimed = [t for t in require.targets if t not in claimed]
    if unclaimed:
        raise CoherenceError(
            f"deliveries/{consumer}.py requires {', '.join(unclaimed)} but no source "
            f"claims {'them' if len(unclaimed) > 1 else 'it'}.\n"
            f"  Open deliveries/{consumer}.py and add "
            f"{unclaimed[0]!r} to the provides= of whichever source carries it.\n"
            f"  Sources: {', '.join(f'{s.level}({s.name})' for s in delivery.send)}.\n"
            f"  This compares REQUIRE against provides= in this file only; it is not "
            f"evidence that the target exists in any run (register C-123)."
        )


def _reconciliation_components(names: list[str]) -> list[set[str]]:
    """Connected components of the reconciliation graph, **among these sources only**.

    ADR-019 §4 says "the declarations *among those sources*", and the previous
    implementation did not honour it: it added an edge to `reconcile_with` whoever that
    was, so two members could be joined transitively through an ensemble the delivery
    never names. Corrected here (#429) because the function is being rewritten anyway and
    the old shape is unreachable from the new rule.

    It also seeded its search at `send[0]`, which was harmless only while every source was
    expected to reconcile. With a coverage source in the list, putting that source first
    made the genuinely reconciled pair read as stranded. Components have no first element.
    """
    members = set(names)
    edges: set[frozenset[str]] = set()
    for name in names:
        meta = source_config(name, "meta")
        partner = meta.get("reconcile_with")
        if meta.get("reconciliation") and partner in members and partner != name:
            edges.add(frozenset((name, partner)))

    components: list[set[str]] = [{name} for name in names]
    for edge in edges:
        touching = [c for c in components if c & edge]
        components = [c for c in components if c not in touching]
        components.append(set().union(edge, *touching))
    return components


def _targets_no_one_else_provides(source, everyone) -> bool:
    """True if every target this source claims is claimed by no other source here.

    ADR-019 §4 (#429): "present **solely** to provide targets no other source in the
    delivery provides". Solely is the operative word — a source that shares one target
    with another and declares no reconciliation with it is the silent-disagreement case
    the reconciliation rule exists to catch, not a coverage source.

    A source with no `provides=` returns False: the question is unanswerable, so the
    stricter branch applies. That is what keeps every delivery written before #427
    behaving exactly as it did.
    """
    if source.provides is None:
        return False
    others = {
        target
        for other in everyone
        if other is not source and other.provides is not None
        for target in other.provides
    }
    return bool(source.provides) and not (set(source.provides) & others)


def _check_reconciliation(delivery, require, consumer: str) -> None:
    """Reconciliation, and the coverage exemption from it (ADR-019 §4, #420 HARD 2).

    **The rule this replaces forbade the only composition that works.** It required every
    source in a delivery to join one connected reconciliation group. `un_crafd` needs three
    targets; every ensemble that reconciles carries one; the only ensemble carrying three
    reconciles with nothing. So the source that supplies the missing targets was refused
    for supplying them.

    Now: every source must **either** join the reconciliation group **or** be present
    solely to provide targets no other source here provides. A source that does neither —
    no stated relationship and no unique targets — is still an error, and that guard is
    unchanged in force.

    What is *not* changed here, deliberately: two or more sources with `reconciled`
    anything other than `True` is still the same hard error (S2, #426). The split governs
    what happens after that gate, not the gate — verified, not assumed, in
    `TestTheSplitDidNotMoveTheGate`. What that leaves unresolved is register **C-145**: a
    delivery combining sources only for coverage must still declare `reconciled=True`,
    which by then claims nothing. Moving the gate is a behaviour change to semantics #426
    pinned four days earlier, and is the maintainer's call.
    """
    if len(delivery.send) < 2:
        return
    names = [s.name for s in delivery.send]
    if require.reconciled is not True:
        raise CoherenceError(
            f"deliveries/{consumer}.py sends {len(names)} sources "
            f"({', '.join(names)}) with reconciled={require.reconciled!r}.\n"
            f"  Open deliveries/{consumer}.py and set reconciled=True in REQUIRE, "
            f"or send one source.\n"
            f"  Several sources with no stated relationship is not currently "
            f"supported; no meaningful use-case has emerged.\n"
            f"  Shipping several sources with no stated relationship silently permits a "
            f"country total that disagrees with the sum of its cells."
        )

    groups = [c for c in _reconciliation_components(names) if len(c) > 1]
    if len(groups) > 1:
        first_of_each = sorted(sorted(g)[0] for g in groups)
        raise CoherenceError(
            f"deliveries/{consumer}.py contains {len(groups)} separate reconciliation "
            f"groups ({'; '.join(', '.join(sorted(g)) for g in groups)}).\n"
            f"  Open deliveries/{consumer}.py and split it into one delivery per group, "
            f"or open {_source_dir(first_of_each[0]).relative_to(REPO_ROOT)}/configs/config_meta.py "
            f"and reconcile the groups with each other.\n"
            f"  Two groups that do not reconcile with each other is the disagreement this "
            f"rule exists to prevent, one level up."
        )
    group = groups[0] if groups else set()

    unattached = [
        s
        for s in delivery.send
        if s.name not in group and not _targets_no_one_else_provides(s, delivery.send)
    ]
    if unattached:
        offenders = {u.name for u in unattached}
        listed = ", ".join(sorted(offenders))
        # The directory named must be the first source *listed*, or the message points
        # at one file while its first sentence points at another.
        directory = _source_dir(sorted(offenders)[0])
        rest = ", ".join(n for n in names if n not in offenders) or "nothing else"
        raise CoherenceError(
            f"deliveries/{consumer}.py sends {listed} — neither reconciled with the rest "
            f"of this delivery ({rest}) nor carrying targets no other source provides.\n"
            f"  Open {directory.relative_to(REPO_ROOT)}/configs/config_meta.py and check "
            f'"reconciliation" and "reconcile_with" — a partner outside this delivery '
            f"does not count.\n"
            f"  Or open deliveries/{consumer}.py and give it a provides= naming the "
            f"targets it alone supplies.\n"
            f"  A source that is neither reconciled nor uniquely needed is a source "
            f"whose disagreement with the others nothing would detect."
        )


def _check_freshness(delivery, require, consumer: str) -> None:
    if delivery.intent.state == "live" and require.max_age is None:
        raise CoherenceError(
            f"deliveries/{consumer}.py is live but declares no max_age.\n"
            f"  Add max_age=months(n) to REQUIRE.\n"
            f"  A live delivery with no freshness bound is how a partner received "
            f"nothing for 145 days while a complete forecast sat on the shelf (#320)."
        )


def _check_tier(delivery, consumer: str) -> None:
    if delivery.tier.name != "prod":
        return
    for source in delivery.send:
        maturity = maturity_of(source.name)
        if maturity != "graduate":
            # ADR-017 §11 "Day-one state": warn, not block, until the real production
            # ensemble is graduated. The migration is not gated on a hasty graduation.
            warnings.warn(
                f"deliveries/{consumer}.py sends '{source.name}', which is "
                f"{maturity}, to a prod consumer. ADR-017 §5 requires graduate.\n"
                f"  Warning rather than failing during the transition "
                f"(ADR-017 §11 day-one state). Not a reason to graduate anything hastily.",
                UserWarning,
                stacklevel=2,
            )


def _check_maturity_rules(delivery) -> None:
    """R1 and R2 (ADR-017 §5), for the ensembles a delivery actually names."""
    for source in delivery.send:
        members = source_config(source.name, "modelset").get("models", [])
        if not members:
            continue
        own = maturity_of(source.name)
        for member in members:
            member_maturity = maturity_of(member)
            if own in ("candidate", "graduate") and member_maturity == "retired":
                raise CoherenceError(
                    f"R1: '{source.name}' is {own} but contains the retired member "
                    f"'{member}'.\n"
                    f"  Open ensembles/{source.name}/configs/config_modelset.py, "
                    f"then models/{member}/configs/."
                )
            if own == "graduate" and member_maturity != "graduate":
                raise CoherenceError(
                    f"R2: '{source.name}' is graduate but its member '{member}' is "
                    f"{member_maturity}.\n"
                    f"  Open ensembles/{source.name}/configs/config_modelset.py, "
                    f"then models/{member}/configs/."
                )


def check(delivery, require, *, consumer: str) -> None:
    """Run every rule that is answerable here. Raises CoherenceError on the first
    violation; warns for the transitional tier rule."""
    _check_resolution_and_level(delivery)
    _check_target_coverage(delivery, require, consumer=consumer)
    _check_reconciliation(delivery, require, consumer)
    _check_freshness(delivery, require, consumer)
    _check_maturity_rules(delivery)
    _check_tier(delivery, consumer)
