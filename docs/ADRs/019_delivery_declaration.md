# ADR-019: The delivery declaration — one file per consumer

**Status:** **Accepted** (2026-08-04) — **amended 2026-08-04** (`live()` → `live(since=…)`, §3);
**amended 2026-08-25** (four documentation corrections from the #420 falsification: §1's example, §3/§7
on `monthly_run.sh`, §4's "where these run", §5's scope — #425. No rule changed.);
**amended 2026-08-25** (`reconciled` documented as three-valued, matching what the checks already do —
§3, §4 — #426. No rule changed.)
**Date:** 2026-08-04
**Deciders:** Simon (maintainer)
**Builds on:** **ADR-017**, which decides that delivery is a `sources → consumer` edge written on the
destination. This ADR decides *what that file looks like*. ADR-017 can stand without this; this cannot
stand without ADR-017.
**Origin:** extracted from ADR-017 §4f/§4d/§5 when that document was split for containment.

---

> **A note on names.** Every model, ensemble, consumer, region and target named in this document is an
> **example**. They are real names where possible, because concrete examples are easier to read than
> placeholders — but which source feeds which consumer changes, consumers are added and retired, and
> buckets get renamed. Nothing here is a declaration about a particular name. The rules are about the
> **shape**; the names are illustration.

---

## Summary

**The problem, in one breath.** ADR-017 says a delivery is an edge written on the destination. It does
not say what the file contains — and the thing it replaces is one line, `"ensemble": "rusty_bucket"`,
buried in `postprocessors/un_fao/configs/config_meta.py`, in a file whose own docstring says
*"This config is for documentation purposes only, and modifying it will not affect the model."* That
sentence is false. That one line decides which forecast reaches the UN.

**The decision, in one breath.** One file per consumer at `deliveries/<consumer>.py`. The **filename
is the consumer**. The body has two blocks: `DELIVERY`, which decides what happens, and `REQUIRE`,
which decides only what is refused.

---

## 1. The file

**An example**, not a specification of a real delivery. `un_ocha`, `slim_chance`, `fat_smooch` and
`land_gaul` are names that could exist, precisely so that nothing here reads as a statement about a
delivery we actually make.

**Both source names are invented (amended 2026-08-25, #425).** This example previously named
`skinny_love`, which is real — and which declares one target, `lr_ged_sb`
(`ensembles/skinny_love/configs/config_meta.py`), while the example asserts three. The note above once
said names are real *"where possible, because concrete examples are easier to read than placeholders"*;
a real name carrying a real constraint the example violates is the one case where that trade-off bites.
The example also does not say **which source provides which target** — a two-source three-target
delivery has no way to express that today. That gap is the subject of #424.

```python
# deliveries/<consumer>.py         <- the filename is the consumer
# shown here as: deliveries/un_ocha.py
from datetime import date

DELIVERY = Delivery(               # DECIDES  — change a line, something different happens
    send      = [pgm("slim_chance"),
                 cm("fat_smooch")],
    frequency = monthly,
    tier      = prod,
    intent    = live(since=date(2026, 8, 4)),
)

REQUIRE = Require(                 # REFUSES  — change a line, a different set is rejected
    reconciled = True,
    targets    = ("lr_ged_sb", "lr_ged_ns", "lr_ged_os"),
    coverage   = "land_gaul",
    max_age    = months(2),
)
```

Read aloud: *"we send slim_chance and fat_smooch to OCHA, monthly, to a production consumer, switched on
since 4 August; they must be reconciled with each other, carry these three targets, cover land_gaul, and
be no older than two months."*

If that sentence is what you were told to do, the file is right. That is the whole test of this design.

## 2. Two blocks, and the difference between them is the point

- **`DELIVERY` decides.** Change a line and something different happens.
- **`REQUIRE` refuses.** Change a line and nothing different is produced — a different set of things
  is *rejected*.

**The rule, and it is testable: removing an *optional* `REQUIRE` line must never change what is
produced, only what is allowed through.** If removing it changes the output, it was a setting and
belongs above.

**The one exception, named rather than buried: `max_age` is mandatory for a `live()` delivery (§4).**
Removing it does not widen what is accepted — it makes the file invalid, so nothing ships. That is a
deliberate asymmetry, not an oversight: a missing freshness bound is the failure that already happened,
five months of it (#320). A rule stated absolutely with an exception a section later is worse than a
rule with its exception attached, because an implementer resolving the contradiction toward "`REQUIRE`
never blocks" would build exactly the gap that caused the incident (register C-128).

This split exists for a specific reason. The file it replaces mixed a setting into a description and
then described itself as inert. Two blocks make that mistake impossible to repeat, because the
question *"does this change anything?"* is answered by which block a key is in.

**`REQUIRE` may be omitted entirely — but only for a `paused()` delivery**, which is the only kind with
nothing mandatory to assert. Every `live()` delivery carries `max_age`, so in practice every delivery
that ships has a `REQUIRE` block. The allowance is narrow and is stated that way rather than as a
general permission, because a general one would read as "this block is optional" and it is not.

The reasoning behind the allowance still stands where it applies: a block that is always present stops
carrying information, and the first person to delete an empty one teaches everyone else to delete
theirs.

## 3. The keys

### Every key, and every value it can take

**Closed** means the allowed values are listed here and anything else is an error. **Open** means the
value is a name checked against something else, so no list can be complete.

| key | block | allowed values | set | what changes if you change it |
|---|---|---|---|---|
| `send` | DELIVERY | a list of `pgm(<source>)`, `cm(<source>)` | closed *(the wrappers)* | **which** forecast the consumer receives |
| `frequency` | DELIVERY | `monthly` | closed — 1 value | **which** scheduled run picks it up |
| `tier` | DELIVERY | `prod` | closed — 1 value | **whether** sources must be `graduate` |
| `intent` | DELIVERY | `live(since=…)`, `paused(reason, since=…)` | closed — 2 values | **whether** it ships at all |
| `reconciled` | REQUIRE | `True`, `False`, unset (`None`) | closed — 3 values | which source *combinations* are refused |
| `targets` | REQUIRE | a tuple of target names | **open** — checked against a run's manifests | a run missing one is refused |
| `coverage` | REQUIRE | one region name | **open** — checked against views-postprocessing | a run with the wrong cell count is refused |
| `max_age` | REQUIRE | `months(n)` | closed *(the wrapper)*, `n` free | an older run is refused — **mandatory when `live()`** |

**Three of the closed sets have exactly one or two values today.** That is stated so it cannot be
mistaken for a rich vocabulary that merely looks small in an example. Each is explained below, with
what would add to it.

**The two open sets are the two checks that leave this repository** — §4's *Where these run*, and
ADR-020 §4's boundaries. They are open for the same reason: neither target names nor region cell
counts are facts this repository owns.

The DELIVERY / REQUIRE split from §2 is visible in the last column: the top four say *which* or
*whether*; every one of the bottom four says *refused*. That is mechanical enough to test.

### The level wrappers inside `send`

| wrapper | means | exists? |
|---|---|---|
| `pgm(<source>)` | the source declares `"level": "pgm"` — grid cell | **yes**, 60 configs |
| `cm(<source>)` | the source declares `"level": "cm"` — country-month | **yes**, 68 configs |
| `admin1(<source>)` | *(not built)* | no admin-1 source exists |

`cm` and `pgm` are the **only** two values `"level"` takes anywhere in this repository today. A third
wrapper arrives with a third source, not before (§3, `send`).

*On `reconciled` being three-valued (amended 2026-08-25, #426).* This row previously said *"closed — 2
values"*. It is three: `Require.reconciled` is declared `bool | None` and **defaults to `None`**, and no
delivery in the platform sets it — `un_fao.py` and `un_crafd.py` both omit the key. So the undocumented
state was the one every real delivery is in. §4 now gives it a rule, and that rule is the one the checks
already implement: unset behaves as `False`. Documented rather than changed — the checks decided this
before the ADR described it, and making the ADR agree is cheaper than a behaviour change nobody asked
for.


**`send` — one or more sources, each with its level claimed.**

`pgm("skinny_love")` does not *set* the level. The level already lives on the ensemble
(`"level": "pgm"`). Writing `pgm(...)` states what you believe, and the system refuses if the source
disagrees. That is why `level` does not appear in `REQUIRE`: it is asserted where you name the source,
which is where a reader looks for it. The shape extends — `admin1(...)` will exist the day an admin-1
source does, and not before.

**Why `send` is a list.** Because ADR-017 §3 decided the delivery edge runs from *sources*, plural: a
consumer may need a grid-cell forecast **and** the country-level forecast it was reconciled against,
and summing grid-cell draws does not reproduce a country-level distribution. **The full argument lives
in ADR-017 §3 and is not repeated here** — it justifies the axis, and this ADR only gives it a syntax.

It also answers the simpler case: a country-level-only delivery names one country-level source —
`send = [cm("<some cm ensemble>")]`. Same key, different source, no schema change.

**`frequency` — required, and today `monthly` is the only value.** There is no safe default. An unlabelled delivery is either picked up by
nothing — a silent non-delivery, the exact failure ADR-017 exists to prevent — or picked up by
everything, so a weekly runner ships monthly products.

Requiring it is also what would let **`monthly_run.sh` stop being a hand-kept list of five paths and
become a filter over declarations** — what runs, and in what order, derived from the same declarations
a human reads rather than typed a second time. Today's order is an unstated data dependency (register
C-122); a filter would remove the need to state it.

**That filter does not exist (amended 2026-08-25, #425).** This paragraph and §7 previously described
it in a tense that read as achieved, while §7's own "Not decided here" said the opposite. Measured:
`monthly_run.sh` contains **zero** references to `deliveries/` or `frequency`, and still ends in a
hand-written block of `run_folder` calls. Worse, `postprocessors/un_crafd` is not among them although
`deliveries/un_crafd.py` is `live()` — so there is a live, armed delivery the monthly path cannot run.
`frequency` makes the filter *possible*; it does not make it *exist*.

*Why a key with one value:* the alternative is no key, and then a second cadence — a weekly internal
run, a quarterly partner — is a schema change touching every existing file rather than one new word.

**`tier` — what kind of consumer this is. One value today: `prod`.**

`prod` means an external partner, or anything the public sees. It does exactly one thing: **every
source must be `graduate`** (ADR-017's maturity axis). That is the whole meaning of the key — it is
the switch that turns the maturity gate on.

*Stated plainly, because it would otherwise read as decided:* with one value, that gate is currently
**unconditional** — every delivery requires graduate sources, so `tier` distinguishes nothing yet. That
is a known temporary state, not a design.

The key exists anyway, for two reasons. ADR-017 already reasons in terms of a *"production-tier
consumer"* (§5), so the concept is in the accepted decision and needs somewhere to live. And a second
value is **blocked on a question ADR-017 deliberately left open** (§12): a `candidate` source writes
nowhere central today, and where a non-production forecast *should* land — a tier-tagged shared shelf,
or a separate one — is not decided. A second tier value cannot be defined before its destination is.

**Named trigger:** *add a second tier value when ADR-017 §12's shadow-destination question is answered.*
It must arrive with a rule of its own. A tier that gates nothing is a label, and labels drift.

**Why it lives on the delivery, not in a separate registry.** With one file per consumer, **the file set
already is the consumer list**: `deliveries/` *is* the register of who we deliver to. A separate
registry would hold exactly one fact the delivery file does not — the tier — so the tier lives here
instead, where it is used.

*What that costs, stated plainly:* there is no edit-time check that a consumer is real.
"Is there an OCHA bucket?" is answered by the platform coordinate registry at run time — one repo away
and later. ADR-020 §4 already records that boundary as a stair ending outside this repository, so this
makes an existing limitation visible rather than adding a new one. If an edit-time check is later
wanted, a registry can return; nothing here forecloses it.

**`intent` — declared; status is derived.** ADR-017 §4e establishes that "in production" is worked out,
never typed. `intent` is the other half: the thing you *do* type, kept in a different word so the two
are never confused. **Both states carry a date; only `paused` carries a reason** — being on is the
default and needs no excuse, whereas switching off silently is the disease being treated.

*The complete set is two values. There is no third, and no bare `intent = live` without the call —
`paused` must carry arguments, so both are constructors for symmetry.*

- **`live(since=…)`** — the scheduled runner picks this delivery up at its `frequency`, and the file
  **must** declare `max_age` (§4). Live is the only state with a freshness obligation, because it is the
  only state where silence is a failure.

  *Why live carries a date too* **(amended 2026-08-04).** This ADR originally had `live()` bare and only
  `paused` dated. But ADR-020 §4 calls the live-but-never-run case *"the hole in the floor"* — a delivery
  declared live that nothing executes raises nothing, because nothing failed. A declared start date does
  not close that hole; it makes it **measurable**. `since` minus the last observed delivery is the length
  of the silence, and without a baseline a brand-new delivery is indistinguishable from one that has been
  dead for five months — which is exactly #320. The usual objection to required dates, that they rot,
  does not apply: `since` is set once at the transition and never updated. It is a fact about a past
  event, not a maintained field.
- **`paused(reason, since=...)`** — the runner skips it. The reason and the date are **required**, and
  they surface in the status report, so a pause is a visible fact with an age rather than an absence.

**`intent` is the arming switch — there is not a second one.** Today the FAO delivery is armed by
`wire_upload_enabled` inside `postprocessors/un_fao/configs/config_meta.py`: views-postprocessing
ADR-013 §11.4 sets `UPLOAD_ENABLED = False` and makes that launcher key its only override. That key
and `intent` are **the same fact**, so the tooling **derives** the launcher key from `intent` rather
than asking anyone to write both. Same principle as the filename carrying the consumer: a thing that
is never typed twice cannot disagree with itself.

This matters more than a tidy-up. Without it, this ADR would move the `"ensemble"` line out of that
file and **leave the on/off switch behind in it** — the very file whose docstring claims to be inert.
That would fix the smell and keep the disease (register C-129).

The disease being treated is two halves quietly waiting with nobody able to see it. A paused delivery
carrying a six-month-old date says so out loud.

*Why not delete the file to turn a delivery off?* Because deleting throws away the reason.
`paused("OCHA bucket not in the registry yet — ask Simon", since="2026-08-04")` is a sentence the next
person can act on. An absent file is not.

### What is deliberately *not* a key

The file this replaces declares eight things. Five map onto the keys above — `name` is the filename,
`level` and `ensemble` are `send`, `targets` and `region` are `REQUIRE`. The other three are named here
so their absence reads as a decision rather than an oversight:

| in the old file | where it goes |
|---|---|
| `wire_upload_enabled: True` | **derived from `intent`** (above) — not a key |
| `wire_contract: True` | **a constant, not a choice.** The legacy leg was retired in #149, so contract mode is unconditional; a key implying it is optional would be false |
| `algorithm: "Postprocessor"` | **stays put** — framework plumbing that tells the pipeline what kind of thing this is. It is not a delivery decision |

## 4. Coherence rules (fail-loud)

These are ADR-017 §5's rules, specialised to the delivery file. ADR-017's maturity rules (R1/R2) are
unchanged and remain there.

- **Resolution.** Every source named must resolve to a real source. The consumer — taken from the
  **filename** — must be a valid identifier; there is no key to disagree with it.
- **Level.** `pgm("x")` fails unless `x` declares `"level": "pgm"`. Neither is authoritative over the
  other; they must agree.
- **Reconciliation.** With `reconciled = True` and two or more sources, the `reconciliation` /
  `reconcile_with` declarations *among those sources* must form **one connected group covering every
  source listed**. A source that reconciles with nothing, or with something outside the delivery, is
  an error naming both files.
  With `reconciled = False` and two or more sources: **hard error** — *"not currently supported; no
  meaningful use-case has emerged."* Shipping several sources with no stated relationship silently
  permits a country total that disagrees with the sum of its cells, which is worse than either source
  alone.
  **Unset (`None`) and two or more sources: the same hard error as `False` (amended 2026-08-25, #426).**
  The reason is the one above — several sources with no stated relationship is the failure, and not
  having said anything is not a statement that they are unrelated. Unset is the *default*, so this is
  the state a two-source delivery lands in by simply not mentioning the key.
  **With one source, `reconciled` is not examined at all** — `True`, `False` and unset are equally
  accepted and none of them means anything. Reconciliation is a property of a *combination*; there is
  no combination to check.
- **Freshness.** A delivery whose `intent` is `live()` **must** declare `max_age`, and refuses to ship
  a run older than it. This is the rule whose absence let a partner receive nothing for five months
  while a complete forecast sat unshipped (#320, C-121).
- **Tier.** A delivery to a `prod` consumer requires every source to be `graduate` (ADR-017 §5).

**Where these run.** Freshness, Level, Reconciliation and Tier are answerable inside this repository at
edit time. `targets` and `coverage` are not — see ADR-020 §4 and §6 below.

**Resolution is answerable here only in part (amended 2026-08-25, #425).** It resolves a *source* name
against `models/` and `ensembles/`, which is in-repo. It does **not** establish that the *consumer* is
real: §3 above says so plainly — *"there is no edit-time check that a consumer is real … answered by the
platform coordinate registry at run time — one repo away and later."* This summary previously read as
though only `targets` and `coverage` left the repository. Three things do.

## 5. Serving-time curation — the approve / quarantine lists

*(Moved unchanged from ADR-017 §4d; it is delivery-side, not axis-side.)*

**This is a per-consumer pattern, not an FAO arrangement (amended 2026-08-25, #425).** It is written
below in FAO's variables because FAO was the first consumer to need it; a second consumer already has
the same mechanism. views-crafdapi defines `APPWRITE_CRAFD_APPROVED_FILE_IDS` and
`APPWRITE_CRAFD_QUARANTINED_FILE_IDS` — `src/views_crafdapi/managers/prediction/quarantine.py`,
documented in its `docs/CICs/PredictionStoreManager.md` — with the same semantics: read at selection
time, unset or empty meaning unrestricted. **A new consumer should expect to need its own pair**, named
for itself, rather than reading this section as something FAO alone has.

Two variables govern *which already-delivered artifacts* the FAO serving layer may return:
`APPWRITE_UNFAO_APPROVED_FILE_IDS` and `APPWRITE_UNFAO_QUARANTINED_FILE_IDS`. Despite the `APPWRITE_`
prefix these are **eligibility data, not connection configuration** — they name which delivered
artifacts are *servable*, an operator decision, not how to reach the store. They therefore belong to
this contract, not to The Appwrite Seam Contract, whose variable map lists them only as explicit
exclusions with a pointer back here (þing-01 verdict D3, 6/6 assent — class is *declared*, never
inferred from a prefix).

- **`APPWRITE_UNFAO_APPROVED_FILE_IDS`** — optional allowlist. When non-empty, only the listed file IDs
  are servable; a newly delivered artifact is **not** served until approved (break-glass; faoapi C-71).
- **`APPWRITE_UNFAO_QUARANTINED_FILE_IDS`** — blocklist. Listed IDs are never served, even if newest —
  how an operator withdraws a bad run.

**Who sets them:** the operator, by editing the deployment environment. They carry non-secret file IDs,
so they are committable and inspectable — never secret. **Who reads them:** views-faoapi at selection
time.

## 6. Open — stated plainly, not smuggled as decided

- **Where the vocabulary lives, and how a delivery file is imported.** The file uses `Delivery`,
  `Require`, `pgm`, `cm`, `monthly`, `live`, `paused`, `months`. Those must come from somewhere.
  views-models is **not** an installable package — `pyproject.toml` holds only pytest markers — and
  today `reconciliation/` is imported by a `sys.path.insert(0, parents[2])` bootstrap in each
  ensemble's `main.py`, with a comment noting that `run.sh` is immutable so `PYTHONPATH` cannot be set
  there.
  The readers of `deliveries/` are **tools and tests**, and both already work from the repo root:
  `python -m tools.liveness` runs today with no install. So no new mechanism is needed for the intended
  readers.
  **Deferred, with a named trigger:** *make views-models an installable package when a `main.py` needs
  to import `deliveries/`.* Today none does — only two `main.py` files import a top-level package at
  all, both for `reconciliation/`. Doing it now would mean adding `pip install -e .` to 131 `run.sh` or
  `requirements.txt` files to benefit two.
- **Whether `admin1(...)` is needed**, and what reconciling three levels means. Not until an admin-1
  source exists.
- **Whether an edit-time consumer check is wanted**, which would bring back a registry (§3).
- **A second `tier` value** — blocked on ADR-017 §12's shadow-destination question, not on this ADR (§3).

## 7. Consequences

**Positive:** "what goes where" collapses from three files in two repositories plus a live bucket query
into one file; a delivery can be read and tested without executing it; `monthly_run.sh`'s hidden
ordering dependency becomes *derivable* — derivable, not derived: see §3 and "Not decided here" below;
a paused delivery cannot be silent.

**Negative:** one more directory to know about. The vocabulary is a small language someone must learn
before writing their first line — mitigated only by the errors ADR-020 requires. And `targets` and
`coverage` are assertions whose checks live outside this repository, so a file that *parses* is not a
delivery that *works* — the tooling must say which kind of check it just ran.

**Transitional, and real:** until `deliveries/` exists, `intent` and `wire_upload_enabled` both exist and
can disagree — and `wire_upload_enabled` is currently present only in an uncommitted working tree
(C-110), so two identical checkouts already publish differently. Deriving one from the other is what
ends that, and it does not end until Phase 1 is built.

**Not decided here:** whether a delivery *runs*. This ADR declares; execution is `monthly_run.sh`
filtering on `frequency`, and that filter does not yet exist.

## 8. Considered alternatives

- **A `to = consumer("un_ocha")` key alongside the filename.** Rejected — two places to state one fact,
  and nothing to reconcile them. ADR-017's own principle is that a thing which is never typed cannot
  lie.
- **`send_pgm` / `send_cm` as separate keys.** Rejected — it names keys after a closed set of levels,
  duplicates the level already on the source, and does not extend to a third level.
- **`reconciled` configured here rather than asserted.** Rejected — reconciliation *changes the
  forecast*, so it belongs to the thing producing it. It is already declared on the ensemble
  (`reconciliation` + `reconcile_with`) with more information than a boolean. A delivery that
  transforms is `postprocessors/` rebuilt under a new name.
- **One central routing table.** Rejected by ADR-017 §9 (alternative B) and still rejected: it
  duplicates membership and becomes a merge bottleneck.

## References

- **ADR-017** — the three axes; this ADR is the file format for its delivery edge.
- **ADR-020** — errors must descend; §4's rules are the worked example of its staircase.
- **views-postprocessing ADR-013** — the wire; owns *how* bytes travel, where this owns *which source
  ships to which consumer*.
- `docs/forecast_delivery_map.md` — what the delivery path actually is today.
- Register: **C-110** (the arming key exists only uncommitted), **C-121** (no age bound),
  **C-122** (order as an unstated dependency), **C-123**/**C-125** (why `targets` cannot yet be an
  edit-time check), **C-126** (live-but-never-run survives this design), **D-09** (is `REQUIRE`
  mandatory).
- views-models **#333** — the second consumer; should arrive as a declaration under this ADR rather
  than as a clone of `postprocessors/un_fao/`.
