# vmo_017 (ADR-017): Forecast Sources, Composition, and Delivery — separating what a model *is*, what it's *built from*, and *where it goes*

**Status:** **Accepted** (2026-07-27) — **revised 2026-08-04**

> **Cite this as `vmo_017` outside this repository.** views-postprocessing and
> views-crafdapi each have their own ADR-017 (*Facts shared with a repository we
> cannot read*, and *Reference Data in Repository*), so a bare "ADR-017" resolves to
> the wrong document for a reader sitting in either of them (#393). The number is
> unchanged and every existing citation stays valid — the prefix is additive.
**Date:** 2026-07-27 (revised 2026-08-04)
**Deciders:** Simon (maintainer)
**Consulted:** platform contributors
**Informed:** all contributors

**Revised 2026-08-04 — split for containment.** This document had grown to ~13 pages and held four
things that change at four different rates. Three moved out, and **no decision was reversed**:

| moved | to | why |
|---|---|---|
| §1, the map of today's system | `docs/forecast_delivery_map.md` | it shrinks as legacy retires; ADRs do not |
| the delivery file format | **ADR-019** | it will be revised as `deliveries/` is used |
| "errors must descend" | **ADR-020** | it governs the whole repo, not delivery |

What stays here is the part that should not need to change: the three axes, derived production status,
the maturity rules, and the shelf write-gate. Per ADR-000 this is a **re-organisation, not a
supersession** — 017 keeps its number and its decisions.


---

> **A note on names.** Every model, ensemble, consumer, region and target named in this document is an
> **example**. They are real names where possible, because concrete examples are easier to read than
> placeholders — but which source feeds which consumer changes, consumers are added and retired, and
> buckets get renamed. Nothing here is a declaration about a particular name. The rules are about the
> **shape**; the names are illustration.

---

## Summary

**The problem, in one breath.** Every model and ensemble carries one hand-typed field, `deployment_status` (one of `{shadow, deployed, baseline, deprecated}`). We ask that single field to answer three unrelated questions at once:

1. how **mature** is this source?
2. what is it **built from**?
3. **where does it go**?

It answers none of them well. And the most important one — *where do the forecasts actually go?* — is written down **nowhere**. That is why, today, turning on a delivery takes the person who built it.

**The decision, in one breath.** Split those three questions into three independent axes, each written in exactly one place:

- **maturity** — on the source;
- **composition** — on the ensemble (unchanged from today);
- **delivery** — a `source → consumer` edge, written on the *destination*.

Then "in production" stops being a label anyone types. It becomes a fact the system *works out* from two things: the source is `graduate`, **and** a delivery ships it to a production consumer. Because nobody writes it by hand, it cannot lie.

**How to read this.** The document is deliberately **bottom-up**. §1 is a concrete map of what happens today; everything after it stands on that map.

---

## 1. How forecast delivery works today

**What this section does:** points at the map, rather than containing it.

How a forecast physically reaches a consumer today — the two stores, the two shelf dialects, the FAO
line, and what is dying — is described in **`docs/forecast_delivery_map.md`**.

That page is deliberately **not** an ADR. It shrinks as legacy retires, and ADR-000 says decisions are
*"never deleted… superseded, not erased"* — so a page designed to change cannot live under this
document's rules. Read it first; everything below stands on it.

## 2. What's broken

**What this section does:** names the specific failures the map produces. Every number here was
measured on 2026-08-04 and names how to re-check it.

- **One label, three jobs.** `deployment_status` mixes *operational mode* (`shadow`/`deployed`),
  *lifecycle* (`deprecated`) and *role* (`baseline`) into one field — three axes, one word.
  **And it is inert.** Nothing anywhere branches `deployed`-vs-`shadow`; that is pinned in both
  repositories by `tests/test_deployment_status_inert.py`, which greps this repo *and* the installed
  `views_pipeline_core` for such a comparison and fails if one appears.
  *(Amended 2026-08-04, when this ADR began to be implemented: there is now **exactly one** declared
  reader — the migration mapping in `deliveries/coherence.py`, which must read the old value in order to
  translate it into maturity per §3. The invariant **tightened rather than lapsed**: every other file
  still fails, and the exemption retires when Phase 2 lands the rename.)*
  *Measured 2026-08-04:* **117 `shadow`, 6 `baseline`, 4 `deprecated`, 1 `deployed`** — 128 files, across
  132 source directories. *(Re-check with a pattern covering **both** quote styles: 81 of the 128 write
  `{'deployment_status': 'shadow'}`, 47 write `{"deployment_status": "shadow"}`. A double-quote-only
  grep reports 47 and silently omits the rest — register C-127.)*
- **The one `deployed` thing in the repository is incoherent.** That single `deployed` source is
  `ensembles/white_mustang`, and both its members — `lavender_haze` and `blank_space` — are `shadow`.
  A deployed ensemble made entirely of things that are not deployed.
- **The rule that should catch that is dead.** pipeline-core's
  `modules/validation/ensemble/check.py:159` reads
  `if single_model_dp_status == "production" and ensemble_deployment_status != "production":` —
  but `production` is not one of the four values this repo writes (`shadow`, `deployed`, `baseline`,
  `deprecated`). The branch cannot fire. The guard has never once run.
- **Delivery is hidden and mislabeled.** The one line controlling FAO delivery sits in
  `postprocessors/un_fao/configs/config_meta.py`, whose docstring says *"This config is for
  documentation purposes only, and modifying it will not affect the model."* The main public line's
  delivery is not written down at all.
- **The shelf has no write-gate.** Anything run with `-p`/`-m` *and* production credentials lands on
  the shelf tagged with its own name — a candidate, a branch fork, a debug run. Nothing structurally
  says "only the real production forecast belongs here." This is how the old store rotted into a pile.
- **Only ensembles can be delivered.** The publish leg takes an `Ensemble`, not a source, so a lone
  model cannot be delivered without wrapping it in a one-member ensemble.
  *Stated as a constraint, not an observation:* no such wrapper exists today — the smallest ensemble in
  the repository has two members. The constraint is real; the symptom has simply not been forced yet.

**Seen in production (2026-07).** Turning the FAO delivery *on* was very hard. Not because forecasts
silently failed — we knew they were not deployed yet. It was hard because *how* to switch it on was
undiscoverable to anyone who had not built it. That is the cost of delivery being declared nowhere.

## 3. The model — three axes, each in one concrete place

**What this section does:** defines the fix — the three axes that replace the one overloaded field.

**The shared abstraction: a *forecast source*.** Anything that emits a forecast is a *source*. A **model** is a *leaf* source; an **ensemble** is a *composite* source (the Composite pattern). Delivery and readiness depend on *source*, never on `Ensemble` specifically — which is why the one-model-ensemble wrapper (§2) disappears.

The three axes, and where each one lives:

- **Maturity** — `candidate → graduate → retired`.
  *Where:* on the source, in `config_maturity.py` (renamed from `config_deployment.py`; same file for models and ensembles). Replaces `deployment_status`.
  *(`baseline` is not a maturity — it's a role, already captured by the algorithm + `regression_point_baselines`. It leaves this file entirely.)*
- **Composition** — an ensemble's members.
  *Where:* `ensembles/<e>/configs/config_modelset.py` — **already exists, unchanged.**
- **Delivery** — a `sources → consumer` edge.
  *Where:* one file per consumer, at `views-models/deliveries/<consumer>.py` — **never on the source**. The **filename is the consumer**; no key repeats it, so the two cannot disagree.
  *(This lifts today's buried `"ensemble"` line into a dedicated, honest file. **ADR-019** gives the format.)*

  **Why *sources*, plural.** A consumer may need a grid-cell forecast **and** the country-level forecast
  it was reconciled against. Both are real products, and one cannot be derived from the other: summing
  grid-cell *draws* does not reproduce a country-level *distribution*, because the sum of marginals is
  not the marginal of the sum unless the joint is preserved.

  This matters here rather than being a statistical aside, because **the platform is
  distribution-native**. Its PredictionFrame ensembles ship uncollapsed pooled draws and consumers
  compute distributional quantities at serve time — highest-density intervals in one case today,
  threshold-exceedance probabilities in another. A consumer computing national uncertainty therefore
  needs the country-level model's own posterior, not a reconstruction of it. That is a claim about the
  *kind* of product this platform ships, not about any particular ensemble or API.

  So the delivery edge names **both** sources. ADR-019 gives that the syntax (`send` takes a list) and
  cites this section rather than restating the argument.

So, in one line: a **model's** config declares only its maturity. An **ensemble's** config declares its
maturity **plus** its members. Neither says anything about delivery or "deployed."

### The three axes as three files

Example names throughout; the shape is the point.

```python
# models/<model>/configs/config_maturity.py        <- AXIS 1: maturity
def get_maturity_config():
    return {"maturity": "graduate"}                #  candidate | graduate | retired
```

```python
# ensembles/<ensemble>/configs/config_modelset.py  <- AXIS 2: composition
def get_modelset_config():
    return {"models": ["<model_a>", "<model_b>"]}   #  unchanged from today
```

```python
# deliveries/<consumer>.py                         <- AXIS 3: delivery
DELIVERY = Delivery(
    send      = [pgm("<ensemble>")],
    frequency = monthly,
    tier      = prod,
    intent    = live(since=…),
)                                                  #  full format: ADR-019
```

Three files, three questions, no overlap. A source never mentions a consumer; a delivery never sets a
maturity; an ensemble never says where its output goes. That separation is the whole decision — and
"is this in production?" is answered by reading the first and third together (§4e), never by a field.

### The values each key can take

| key | file | allowed values | set |
|---|---|---|---|
| `maturity` | `config_maturity.py` | `candidate`, `graduate`, `retired` | closed — 3 values |
| `models` | `config_modelset.py` | model directory names | open — must resolve under `models/` |
| `level` | `config_meta.py` | `cm`, `pgm` | closed — 2 values |
| `reconciliation` | `config_meta.py` | `"pgm_cm_point"`, `None` | closed — 2 values |
| `reconcile_with` | `config_meta.py` | a source name | open — required when `reconciliation` is set |
| the delivery keys | `deliveries/<consumer>.py` | — | **ADR-019 §3** |

**`maturity`'s three values replace four in use today — and the migration is decided here.** The field
being replaced (`deployment_status`) holds `shadow`, `baseline`, `deployed`, `deprecated`; §2 records
that only `deprecated` does anything.

| today | becomes | note |
|---|---|---|
| `shadow` | `candidate` | the bulk of the fleet |
| `deprecated` | `retired` | the only value with behaviour today |
| `baseline` | `candidate` | the **role** leaves this file; the source still needs a maturity |
| `deployed` | `graduate` **only if R2 holds**, else `candidate` | see below |

**Two groups the table alone does not cover.** The six `baseline` sources keep their *role* — it already
lives in the algorithm plus `regression_point_baselines`, which is why it leaves this file — and take
`candidate` as their maturity, because nothing about being a baseline makes a source eligible to ship.
And four source directories have **no `config_deployment.py` at all** (`models/cool_cat`,
`models/teenage_dirtbag`, `models/test_model`, `ensembles/test_ensemble`), so they have nothing to
migrate *from*; they get `config_maturity.py` created with `candidate`.

**Why `deployed` is not a straight rename.** Mapped naively it breaks R2 immediately: measured
2026-08-04, exactly one source is `deployed` — an ensemble whose two members are both `shadow`. A
straight rename makes it a `graduate` ensemble with `candidate` members, so **the migration would
produce a violation of this ADR's own rule on the day it lands**. Hence the conditional: a `deployed`
source becomes `graduate` only where its members already qualify, and `candidate` otherwise.

Nothing is lost by that. `graduate` does not mean *is delivered* — it means *eligible* to be, and
whether it is delivered is the third file's business. Today's single `deployed` source has no delivery
edge at all, so demoting it to `candidate` changes no behaviour; it only stops the label asserting a
readiness the members do not have.

**`level` has exactly two values** across all 128 configs that declare it. This is what ADR-019's
`pgm(...)` / `cm(...)` wrappers check against, and why there is no third wrapper.

## 4. How you actually operate it

**What this section does:** the practical part — how you put something into production, and how a forecast physically flows. Everything here names real files.

### 4a. Put a source into production — the recipe
"In production" means *delivered to one or more API endpoints*, and an endpoint is a **consumer**. You never flip a status — you take these steps:

- **A single model → an endpoint:**
  1. `maturity: graduate` in the model's config;
  2. a file `deliveries/<endpoint>.py` whose `send` names that model;
  3. that file declares `tier = prod` (ADR-019 §3).
- **An ensemble → an endpoint:** identical — `send` names the ensemble.
- **A grid-cell ensemble reconciled against a country-level one → an endpoint:** `send` names **both**, and `REQUIRE.reconciled = True`. The pairing itself is not restated here; it already lives on the ensembles (`reconciliation` + `reconcile_with`), and §5's reconciliation rule checks that what you listed matches what they declare.
- **A model, inside an ensemble → an endpoint:** add it to the ensemble's `config_modelset.py`; make both the model and the ensemble `graduate`; declare the ensemble's delivery edge. The model is now in production *transitively*.

### 4b. How a forecast reaches the shelf — three guards
A forecast lands on the **production shelf** only if **all three** of these hold. Each is checked where its information already lives:

- **Intent — the `-p` flag** (per run). Defaults off. Without it, a run writes nothing to the shelf — so you can run production locally to inspect it *without* publishing. *(Keep this flag.)*
- **Eligibility — `maturity == graduate`** (the source's own config). A `candidate` **cannot** write to the prod shelf. This is the write-gate that's missing today, and it kills the pollution structurally — no consumer knowledge required.
- **Credentials** — the production `.env`.

**Candidates go nowhere central, for now** — they stay on the machine that ran them. *(Deferred: a shared "shadow shelf" for scheduled candidate runs — see §12.)*

### 4c. How a forecast is served — split by boundary
The producer must **never** know its consumers. So the gate is split across two boundaries, and neither side reads the other's config:

- **Write → shelf** is gated by **maturity** (§4b) — the source's own config.
- **Shelf → consumer** is gated by the **delivery declaration** — the delivery unit reads its *own* `config_delivery.py`, pulls only its declared source off the shelf (by declared identity, not by filename), and serves it.

### 4d. Serving-time curation

Which *already-delivered* artifacts a consumer may serve (the FAO approve / quarantine lists) is
delivery-side, not axis-side. It moved to **ADR-019 §5**.

The two meet only at the shelf — which now holds only `graduate` forecasts. A graduate-but-undelivered
forecast sitting there is fine: it is finished, just not routed anywhere yet.

### 4e. "In production" is derived, never declared
> A source is *in production* ⟺ its maturity is `graduate` **and** a delivery ships it (directly, or via a composite that contains it) to a **production-tier** consumer.

Nobody types "deployed." "Is this in production?" is worked out on demand from those two facts, and never stored — so it can't lie, because there's no field to lie in.

*One honest caveat about the second condition.* `tier` currently has exactly **one** value, `prod`
(ADR-019 §3), so "to a **production-tier** consumer" is today equivalent to "at all". The definition is
written for the general case and becomes discriminating the moment a second tier value exists — which
is itself blocked on §12's open shadow-destination question. Until then, do not read the qualifier as a
check that is running.

*Analogy:* a sticky note reading *"light: ON"* can be wrong; but *checking that there's a working bulb **and** the switch is wired and flipped* cannot.

### 4f. What a delivery file looks like

**ADR-019** decides the format: one file per consumer at `deliveries/<consumer>.py`, the filename
carrying the consumer's identity, and the body split into what the delivery *does* and what it
*requires*.

It is a separate ADR because it will be revised as `deliveries/` is built and used, while the three
axes above should not need to change at all.

## 5. Coherence rules (fail-loud)

**What this section does:** the checks that keep the model honest. Each fails loudly rather than letting a bad state pass quietly.

**Maturity rules** (need only the source configs — so they're checked early, at config-load, by the sniffer):

- **R1:** no *active* ensemble (`candidate` or `graduate`) may contain a `retired` member.
- **R2:** a `graduate` ensemble's members must **all** be `graduate`. *(This is the dead `white_mustang` rule, revived.)*

**Delivery rules:**

- **Tier rule** (needs the delivery edge — so it is checked at the delivery boundary): a delivery to a
  **production-tier** consumer requires its sources to be `graduate`.
- **Resolution rule:** every source a delivery names must resolve to a real source.

The rules specific to the delivery *file* — level claims, the reconciliation graph, freshness — are in
**ADR-019 §4**, because they change with the format rather than with the model.

The delivery boundary is the authoritative gate; the config-load checks just shorten the feedback loop. Because R1/R2 need no delivery knowledge, an incoherent ensemble is caught even while it sits undelivered.

## 6. Relationship to vpp ADR-013

**What this section does:** places this ADR next to its sibling in views-postprocessing, so the boundary between them is clear.

**views-postprocessing's ADR-013** (the Sampled-Forecast Wire Contract, Accepted 2026-07-15) and **this ADR** (views-models ADR-017) are companions that split one territory:

- **vpp-013 owns the wire** — *how* forecast bytes travel (formats, manifests, the pinned consumer `name`).
- **ADR-017 owns the relationship** — *which* source ships to *which* consumer.

Like a parcel: **vpp-013 standardises the packaging; ADR-017 writes the address.** Neither replaces the other. And vpp-013 is what makes the shelf addressable by *declared provenance* instead of fragile filename (§1) — which is the mechanism that lets a delivery declaration actually find its source.

## 7. The derived state has an instrument

"Derived from what, verified how?" The declaration side is §4e. The *verification* side is code: `tools/liveness` (epic #238) observes every delivery surface — the shelf, `unfao_bucket`, the public API — with raw facts.

So "in production" is checkable end-to-end: **declared** (a delivery edge exists) **and observable** (its liveness surface is green). A declared-but-stalled delivery shows up as exactly that, instead of a label nobody can falsify.

## 8. Rationale (against the maintainer's principles)

- **SRP** — an ensemble changes only for composition; delivery only in the delivery layer; maturity only on the source.
- **DIP / ISP / LSP** — delivery depends on the narrow `forecast source` interface, so a lone model can stand in for an ensemble and the wrapper hack becomes impossible.
- **OCP** — a new consumer is just a new delivery edge; the status enum stops being an edit-point.
- **ADP** — delivery depends on the source, not the reverse; the producer never gains a back-edge to its consumers (§4c).
- **Screaming architecture** — an explicit delivery layer makes "where forecasts go out" *visible*, instead of emergent across a bash list and a dead label.

## 9. Considered alternatives

- **A — keep `deployment_status`, add a `destinations` field on the source.** This makes the producer know its consumers (an ADP/SRP violation). **Rejected.**
- **B — one central routing config (an ensemble → consumer map).** Duplicates membership and becomes a merge bottleneck. **Rejected as the source of truth** — a *derived* topology view gives the same at-a-glance benefit without the drift.
- **C — derive routing from status alone.** Can't express "delivered to FAO but not to the main API." **Rejected.**

## 10. Consequences

**Positive:**

- `deployed` can't lie (it's derived);
- the `white_mustang` class of incoherence is caught automatically;
- a lone model is directly deliverable — no wrapper;
- the shelf holds only `graduate` forecasts;
- delivery topology is explicit and can be generated, not hand-kept;
- each config file answers exactly one question.

**Costs:** the `deployment_status → maturity` rename is a **cross-repo breaking change** against a *published* pipeline-core (the C-73 / C-132 skew tax). The full set of things that must move together:

- views-models configs;
- pipeline-core: the sniffer, the templates, the ensemble guard;
- the ensemble-only publish leg (`sampled_forecast_publisher`, pipeline-core #269) — it must accept a *source*, not just an `Ensemble`;
- the silent log-stamp default `c.get("deployment_status", "shadow")`;
- ADR-003.

Also: making the main line an explicit delivery unit adds new structure on the most operationally sensitive path.

## 11. Implementation (phased — not big-bang)

- **Phase 0 (this ADR):** the shared model. Zero code.
- **Phase 1 — cheap, local, breaks nothing published:** revive the dead guard (R2); create `deliveries/` (ADR-019) with **`un_fao.py` written first as a description of what already runs** — a characterisation, not a change — then lift FAO's `"ensemble"` line out of `config_meta.py`; derive `is_in_production`. *(Sequencing: do this **before** views-models#333 clones `postprocessors/un_fao/` into a second consumer directory, so consumer number two arrives as a declaration rather than as a second copy of an invisible edge.)*
- **Phase 2 — cross-repo, deliberate:** the `deployment_status → maturity` rename + value remap + the pipeline-core contract + ADR-003; rename the file `config_deployment.py → config_maturity.py`; delete the silent log-stamp default.
  Do this with a **dual-vocabulary transition window** — the sniffer accepts both old and new values, warns on the old, and flips to new-only in a later major (the gid→id playbook). It **cannot** ride the pipeline-core 3.0 release, so it lands as a post-3.0 major or its own coordinated bump.
- **Phase 3 — structural:** make the main public line an explicit delivery unit; add the **shelf write-gate** (only `graduate` writes).
- **Phase 4:** re-home the ensemble guard — **by moving its function, not deleting it.** Its live `deprecated`-member check is the *only* ensemble-time member-status check (the sniffer never sees member configs), so deleting it outright would remove real coverage.

**Day-one state (known, temporary).** On adoption, the platform inherited exactly one coherence violation: the placeholder `rusty_bucket` (`candidate`) delivers to the production-tier `fao` consumer — a pre-production shakedown. **As of 2026-08-11 there are two of the same kind**: `un_crafd` (#333) delivers the same `candidate` ensemble to the production-tier CRAF'd consumer. One source, two edges — the count changed, the situation did not, and it resolves the same way, by graduating `rusty_bucket`. During the transition, the tier-rule check **warns, not blocks**, on this edge, until the real production ensemble is graduated. The migration is *not* gated on a hasty graduation.

## 12. Decided vs Deferred/Open

**Decided (this review, 2026-07-27):**

- Only `graduate` sources write to the prod shelf; **candidates go nowhere central, for now.**
- Three shelf-write guards: `-p` intent / `graduate` eligibility / credentials. **Keep `-p`.**
- Gate split by boundary: maturity gates the write, delivery gates the serve; the producer never reads consumer config.
- Maturity rules **R1** (no retired member in an active ensemble) + **R2** (graduate ensemble → all-graduate members).

**Decided — the delivery unit** (the detail is ADR-019):

- **Delivery-unit home: `deliveries/<consumer>.py`.** Decided on legibility rather than architecture — the architecture was indifferent between this and `postprocessors/`, but the person filling one in is not. They arrive asking *"how do I send this to them?"*, not *"where are the postprocessors?"* — and `un_fao` does not post-process anything, it delivers.
- **Delivery-unit weight: lightweight.** A delivery **declares**; it never transforms. If a delivery file ever grows transformation logic we have rebuilt `postprocessors/` under a new name.

The file's shape, its keys and its rules are ADR-019's.

**Deferred / open (stated plainly — not smuggled as decided):**
- **Scheduled candidate / shadow ensembles** (e.g. a `views_shadow` endpoint run monthly): shared prod shelf tier-tagged, or a separate shadow shelf? *Revisit soon.*
- **Two stores today** (the legacy `views-forecasts` store for the public API + the Appwrite `production_forecasts` shelf): eventual consolidation, and the maintainer's "on Hetzner one day" intent — not now.
- **`config_meta.py` cleanup:** the "documentation only" docstring must stop being a lie once delivery moves out.
- **The guard's stale-log gap:** even the live `deprecated`-member check reads the member's *artifact-time log*, so a member deprecated *after* its artifacts were generated slips through (register-worthy, Tier-3).

## 13. Errors

When a config here is wrong, the error must send the reader **exactly one level down**, naming the
next file — and where the reader cannot go further, name a person rather than a task they cannot
perform. That rule governs the whole repository, not just delivery, so it is **ADR-020**.

## References

- **ADR-019** (the delivery declaration) and **ADR-020** (errors must descend) — split out of this document 2026-08-04.
- **`docs/forecast_delivery_map.md`** — how delivery works today; this ADR's §1 lives there.
- ADR-001 (ontology), ADR-002 (topology), ADR-003 (authority — the `deployment_status` allowed-list), ADR-016 (the same derive-don't-declare instinct).
- views-postprocessing **ADR-013** (the wire half; see §6).
- pipeline-core **Lean Platform End-State Roadmap** (`documentation/plans/2026-07-27_lean_platform_end_state_roadmap.md`) — owns the sequencing/retirement this ADR's §1 direction-of-travel defers; the basis for this ADR's acceptance.
- `reports/expert_reviews/2026-07-19_adr013_wire_contract_review_views_models_seat.md`.
- `tests/test_deployment_status_inert.py` (pins the "label is inert" fact from §2).
- `tools/liveness` / epic #238 (the instrument in §7).
- pipeline-core: `modules/validation/core_config_sniffer.py`; `modules/validation/ensemble/check.py` (the dead guard); `managers/prediction/io.py` (gen-1: one method, two stores, `type=target`); `managers/model/model.py:572` (the gen-2 conditional saver trio); `cli/args.py` (`-m`); **`tests/test_managers/test_delivery_characterization.py`** (pins the three producer generations + the two shelf dialects).
- Session design discussion, 2026-07-02; grounding revision, 2026-07-27.
