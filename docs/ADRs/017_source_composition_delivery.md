# ADR-017: Forecast Sources, Composition, and Delivery — separating what a model *is*, what it's *built from*, and *where it goes*

**Status:** **Accepted** (2026-07-27)
**Date:** 2026-07-02 — **revised 2026-07-27** (grounded top-to-bottom in the real cross-repo flow; operational mechanics written from scratch; decided-vs-open made explicit. The core decision is unchanged.)
**Accepted on the basis that** it aligns with pipeline-core's *Lean Platform End-State Roadmap* (`views-pipeline-core/documentation/plans/2026-07-27_lean_platform_end_state_roadmap.md`), which owns the sequencing/retirement this ADR's §1 direction-of-travel defers. The two documents adopt each other's vocabulary and read as one system.
**Deciders:** Simon (maintainer) — Accepted 2026-07-27
**Consulted:** platform contributors
**Informed:** all contributors

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

## 1. How forecast delivery works *today* — the concrete map

**What this section does:** shows you, with real file paths, how a forecast actually gets from a model run to an API today — so the rest of the ADR has solid ground under it.

Two things make today's picture harder than it should be:

- there are **two stores** — two different central places a forecast can land; and
- there are **three producer mechanisms**, from three eras of the codebase, all alive at the same time. Which one runs depends on what shape a run produces: a pandas DataFrame, a PredictionFrame, or a PredictionFrame-ensemble.

(The full mechanics are pinned in pipeline-core `tests/test_managers/test_delivery_characterization.py`. Here we keep only what the *delivery* story needs.)

This is a lot of accumulated legacy — there's no pretending otherwise. To keep it readable, every piece below carries a tag: **[LEGACY]** (alive only until retired), **[CURRENT]** (the working present), or **[TARGET]** (where everything is converging). And §1 deliberately *ends* on the direction of travel — so you leave this section seeing a transition, not a permanent mess.

> **Two words, kept distinct throughout this document.** Both are "stores" in plain English, which is exactly what trips people up:
> - **"store"** = the **OLD** `views-forecasts` central store — feeds the public API, pandas-only door, **[LEGACY]** (the "pile nobody opens").
> - **"shelf"** = the **NEW** Appwrite `production_forecasts` bucket — feeds FAO, metadata-tagged, **[CURRENT]**.
>
> They are two different places. Wherever this ADR says *store* it means the old one; *shelf* means the new one.

### The monthly run — this is what actually ships today. `[LEGACY]`

`monthly_run.sh` is a hand-run list. It runs 4 **legacy DataFrame ensembles** (`pink_ponyclub`, `skinny_love`, `rude_boy`, `first_love`), each with `-m`.

(`-m` = `--monthly`; it bundles train + forecast + report + prediction_store.)

Each run goes through **one** method — `_save_predictions` → `PredictionIOManager`. (Not a set of "saver" objects — that's a different path, below.) That one method writes the forecast three ways:

```
monthly ensemble run (-m)  ->  PredictionIOManager        [LEGACY]
  ├─ local disk: pandas-parquet (legacy list-in-cell)
  ├─ legacy "views-forecasts" store  (df.forecasts.to_store)   [LEGACY]
  │    -> external API  api.viewsforecasting.org   [MAIN PUBLIC LINE]
  │       (prio-data/views_api, external)
  └─ Appwrite SHELF: production_forecasts,  type="ensemble"    [LEGACY dialect]
```

There *is* a "savers" path — the single-model PredictionFrame path, with the composed `LocalParquetSaver` / `ViewsForecastsSaver` / `AppwriteSaver` trio (model.py:572). But that is a **different, conditional** mechanism `[CURRENT]`, and it is **not** what the monthly ensembles use.

### `rusty_bucket` is a different animal again. `[TARGET]`

`rusty_bucket` is a **PredictionFrame ensemble (PFE)** — the shape everything is converging toward. It uses *neither* path above. It does two things:

- writes `save_pf` (npy/npz) to local disk; and
- with the store on, publishes **wire shards** to the *same shelf*, tagged **`type="sampled_forecast_*"`**.

Those wire shards are defined by **views-postprocessing's ADR-013** (its Sampled-Forecast Wire Contract). This is the *contract dialect*, kept deliberately separate from the legacy `type="ensemble"` dialect.

> *Note on numbering:* throughout this document, **"ADR-013" always means views-postprocessing's ADR-013** (the wire contract) — a *different repo's* ADR. It is **not** this repo's ADR-013 (`013_regression_target_name_agnosticism.md`). Written **vpp ADR-013** where confusion is likely.

### So the one shelf holds two disjoint dialects

- legacy documents tagged `type="ensemble"`, and
- contract shards tagged `type="sampled_forecast_*"`.

They never overlap. That separation is the vpp ADR-013 §11.4 transition invariant.

### The FAO line has two legs, and they share that one shelf

```
LIVE legacy leg:  shelf type="ensemble"  ->  vpp UNFAO manager
    filters {category:forecast, type:ensemble}   (unfao.py:35)
    reads "ensemble":"rusty_bucket"              (unfao.py:109)
    enrich (GAUL) -> validate -> unfao_bucket, name="un_fao"
    gated: UPLOAD_ENABLED = False
      ->  views-faoapi serves unfao_bucket, name="un_fao"

DORMANT contract leg:  shelf type="sampled_forecast_*"
    ->  vpp wire/source_selection: newest manifested run
    ->  same enrich / validate -> same unfao_bucket
```

### The punchline — and it's exactly what this ADR is about

Put those pieces together:

- the **live** FAO leg reads `type="ensemble"`;
- but the **declared** FAO source (`rusty_bucket`) is a PFE, so it only ever produces `type="sampled_forecast_*"`;
- therefore **the live leg will never see `rusty_bucket`'s output.**

Meanwhile, the 4 monthly ensembles *do* produce `type="ensemble"` — but they aren't the declared FAO source.

So today's FAO delivery is a **live consumer leg pointed at a source that isn't scheduled, while that source's own output can only feed the leg that's switched off.** Both halves are quietly *waiting*. And you cannot see this crossed-wires state by reading any single config file. That invisibility is the whole cost of "no single place declares delivery."

### Where each config lives today (real paths)

- **A model's maturity-ish label:** `views-models/models/<m>/configs/config_deployment.py` → `deployment_status`.
- **An ensemble's label + members:** `.../ensembles/<e>/configs/config_deployment.py` (the label) and `.../config_modelset.py` (the members).
- **The FAO "which source feeds us" declaration:** `views-models/postprocessors/un_fao/configs/config_meta.py` — the single line `"ensemble": "rusty_bucket"`, read by the manager at `views-postprocessing/.../unfao/managers/unfao.py:109`.
  - **The smell, exactly:** that file's own docstring says *"This config is for documentation purposes only, and modifying it will not affect the model."* That is **false** — that one line decides which forecast reaches the UN.
- **The main public line's declaration:** *none exists.* It is emergent — `monthly_run.sh` + the legacy store + the external API.

### Two stores, two roles

- **The `views-forecasts` store** (the old one) — pandas-only front door (`df.forecasts.to_store`); feeds the public API. Unstructured, matched by fragile naming — the pile nobody opens. *(It also has two **non-delivery** roles this ADR doesn't govern — legacy-ensemble constituent transport, and a run-metadata registry — whose retirement is sequenced by the pipeline-core roadmap, §"Direction of travel".)*
- **The Appwrite `production_forecasts` shelf** (the new one) — the two-dialect bucket above; feeds the FAO postprocessor. vpp ADR-013 makes its contract dialect addressable by *declared provenance* instead of filename.

### Direction of travel — what's dying, and where it goes

*(Context only. This convergence is owned by the views-frames migration + vpp ADR-013, and **sequenced by pipeline-core's Lean Platform End-State Roadmap** (2026-07-27) — **not decided by this ADR**. It's here so the mess above reads as transitional.)*

- **DataFrame ensembles** (the 4 monthly) → **PredictionFrame / PFE** (the `rusty_bucket` shape).
- **the `views-forecasts` store** (pandas-only door) → **the Appwrite shelf** (→ Hetzner, eventually).
- **shelf dialect `type="ensemble"`** → **`type="sampled_forecast_*"`** (vpp ADR-013 §11.4).
- **the legacy FAO leg** → **the contract FAO leg**.

Everything converges on one shape: **frames-native producers → the shelf's contract dialect → contract-reading consumers.** This ADR is written for that **target** state. The legacy machinery is *grandfathered* — described here so it's visible, not endorsed. **When a legacy element retires, its `[LEGACY]` line here retires with it** — so this section is a shrinking list, by design.

## 2. What's broken

**What this section does:** names the specific failures the map above produces.

- **One label, three jobs.** `deployment_status` mixes *operational mode* (`shadow`/`deployed`), *lifecycle* (`deprecated`), and *role* (`baseline`) into one field — three different axes, one word. And it's almost **inert**: only `deprecated` actually does anything (the config sniffer refuses to run it). `deployed`, `shadow`, and `baseline` are indistinguishable on a run. *(Verified: no code anywhere branches `deployed`-vs-`shadow`; pinned by `tests/test_deployment_status_inert.py`.)*
- **Delivery is hidden and mislabeled.** The one line that controls FAO delivery sits inside a config that claims to do nothing (§1). The main line's delivery isn't written down at all.
- **The shelf has no write-gate.** Anything run with `-p`/`-m` *and* the production credentials lands on the shelf, tagged with its own name — a `candidate`, a branch fork, a debug run. Nothing structurally says "only the real production forecast belongs here." This is how the old store rotted into a pile.
- **The one coherence rule that should catch trouble is dead.** pipeline-core's ensemble guard only fires on the impossible value `"production"`, so a `deployed` ensemble can silently contain `shadow` members — and `white_mustang` already does (`lavender_haze`, `blank_space`).
- **Delivering a lone model needs a fake ensemble.** Only ensembles can be delivered, so a single model gets wrapped in a one-member "ensemble" — the tell of a missing shared abstraction.

**Seen in production (2026-07).** Turning the FAO delivery *on* was very hard. Not because forecasts silently failed — we knew they weren't deployed yet. It was hard because *how* to switch it on was undiscoverable to anyone who hadn't built it. That is the cost of delivery being declared nowhere.

## 3. The model — three axes, each in one concrete place

**What this section does:** defines the fix — the three axes that replace the one overloaded field.

**The shared abstraction: a *forecast source*.** Anything that emits a forecast is a *source*. A **model** is a *leaf* source; an **ensemble** is a *composite* source (the Composite pattern). Delivery and readiness depend on *source*, never on `Ensemble` specifically — which is why the one-model-ensemble wrapper (§2) disappears.

The three axes, and where each one lives:

- **Maturity** — `candidate → graduate → retired`.
  *Where:* on the source, in `config_maturity.py` (renamed from `config_deployment.py`; same file for models and ensembles). Replaces `deployment_status`.
  *(`baseline` is not a maturity — it's a role, already captured by the algorithm + `regression_point_baselines`. It leaves this file entirely.)*
- **Composition** — an ensemble's members.
  *Where:* `ensembles/<e>/configs/config_modelset.py` — **already exists, unchanged.**
- **Delivery** — a `source → consumer` edge.
  *Where:* a delivery unit's `config_delivery.py` (e.g. `views-models/postprocessors/un_fao/configs/config_delivery.py`) — **never on the source**. Each consumer is also registered once, with a tier, in `views-models/meta/consumers.py`.
  *(This lifts today's buried `"ensemble"` line into a dedicated, honest file.)*

So, in one line: a **model's** config declares only its maturity. An **ensemble's** config declares its maturity **plus** its members. Neither says anything about delivery or "deployed."

## 4. How you actually operate it

**What this section does:** the practical part — how you put something into production, and how a forecast physically flows. Everything here names real files.

### 4a. Put a source into production — the recipe
"In production" means *delivered to one or more API endpoints*, and an endpoint is a **consumer**. You never flip a status — you take these steps:

- **A single model → an endpoint:**
  1. `maturity: graduate` in the model's config;
  2. a delivery unit's `config_delivery.py` = `{"source": "<model>", "consumer": "<endpoint>"}`;
  3. `<endpoint>` registered in `meta/consumers.py` with `tier: prod`.
- **An ensemble → an endpoint:** identical, with `"source": "<ensemble>"`.
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

### 4d. Serving-time curation — the FAO approve / quarantine lists
Two variables govern *which already-delivered artifacts* the FAO serving layer (views-faoapi) may return: `APPWRITE_UNFAO_APPROVED_FILE_IDS` and `APPWRITE_UNFAO_QUARANTINED_FILE_IDS`. Despite their `APPWRITE_` prefix, these are **eligibility data, not connection configuration** — they name which delivered artifacts are *servable*, an operator/curation decision, not how to reach the store. They therefore belong to **this** eligibility contract, not to the identity/secrets/config contract (PLATFORM-001), whose variable map lists them only as explicit exclusions with a pointer back here (þing-01, verdict D3; 6/6 assent — class is *declared*, never inferred from a prefix).

- **`APPWRITE_UNFAO_APPROVED_FILE_IDS`** — an optional allowlist. When set (non-empty), *only* the listed Appwrite file IDs are servable — a newly delivered artifact is **not** served until explicitly approved (a break-glass gate; faoapi C-71). When empty/unset, selection is unrestricted and the newest fully-manifested run wins.
- **`APPWRITE_UNFAO_QUARANTINED_FILE_IDS`** — a blocklist. Listed file IDs are never served, even if newest — how an operator withdraws a bad run.

**Who sets them:** the operator (the human who holds the keys, PLATFORM-001 §operator), by editing the deployment environment. They carry non-secret Appwrite file IDs, so they are committable/inspectable — never secret. **Who reads them:** views-faoapi at selection time. With manifest-first serving (faoapi #263) these lists are a *break-glass* control layered over the newest-wins default, not the primary selection mechanism.

They meet **only at the shelf** — which now holds only `graduate` forecasts. A graduate-but-undelivered forecast sitting there is fine: it's finished, just not routed anywhere yet.

### 4d. "In production" is derived, never declared
> A source is *in production* ⟺ its maturity is `graduate` **and** a delivery ships it (directly, or via a composite that contains it) to a **production-tier** consumer.

Nobody types "deployed." "Is this in production?" is worked out on demand from those two facts, and never stored — so it can't lie, because there's no field to lie in.

*Analogy:* a sticky note reading *"light: ON"* can be wrong; but *checking that there's a working bulb **and** the switch is wired and flipped* cannot.

## 5. Coherence rules (fail-loud)

**What this section does:** the checks that keep the model honest. Each fails loudly rather than letting a bad state pass quietly.

**Maturity rules** (need only the source configs — so they're checked early, at config-load, by the sniffer):

- **R1:** no *active* ensemble (`candidate` or `graduate`) may contain a `retired` member.
- **R2:** a `graduate` ensemble's members must **all** be `graduate`. *(This is the dead `white_mustang` rule, revived.)*

**Delivery rules:**

- **Tier rule** (needs the delivery edge — so it's checked at the delivery boundary): a delivery to a **production-tier** consumer requires its source to be `graduate`.
- **Resolution rule:** a delivery's `source` and `consumer` must both resolve — a real source, and a consumer in the registry.

The delivery boundary is the authoritative gate; the config-load checks just shorten the feedback loop. Because R1/R2 need no delivery knowledge, an incoherent ensemble is caught even while it sits undelivered.

## 6. Relationship to vpp ADR-013

**What this section does:** places this ADR next to its sibling in views-postprocessing, so the boundary between them is clear.

**views-postprocessing's ADR-013** (the Sampled-Forecast Wire Contract, Accepted 2026-07-15) and **this ADR** (views-models ADR-017) are companions that split one territory:

- **vpp-013 owns the wire** — *how* forecast bytes travel (formats, manifests, the pinned consumer `name`).
- **ADR-017 owns the relationship** — *which* source ships to *which* consumer.

Like a parcel: **vpp-013 standardises the packaging; ADR-017 writes the address.** Neither replaces the other. And vpp-013 is what makes the shelf addressable by *declared provenance* instead of fragile filename (§1) — which is the mechanism that lets a delivery declaration actually find its source.

## 7. The derived state has an instrument

"Derived from what, verified how?" The declaration side is §4d. The *verification* side is code: `tools/liveness` (epic #238) observes every delivery surface — the shelf, `unfao_bucket`, the public API — with raw facts.

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
- **Phase 1 — cheap, local, breaks nothing published:** revive the dead guard (R2); lift FAO's `"ensemble"` line out of `config_meta.py` into `config_delivery.py`; add `meta/consumers.py`; derive `is_in_production`.
- **Phase 2 — cross-repo, deliberate:** the `deployment_status → maturity` rename + value remap + the pipeline-core contract + ADR-003; rename the file `config_deployment.py → config_maturity.py`; delete the silent log-stamp default.
  Do this with a **dual-vocabulary transition window** — the sniffer accepts both old and new values, warns on the old, and flips to new-only in a later major (the gid→id playbook). It **cannot** ride the pipeline-core 3.0 release, so it lands as a post-3.0 major or its own coordinated bump.
- **Phase 3 — structural:** make the main public line an explicit delivery unit; add the **shelf write-gate** (only `graduate` writes).
- **Phase 4:** re-home the ensemble guard — **by moving its function, not deleting it.** Its live `deprecated`-member check is the *only* ensemble-time member-status check (the sniffer never sees member configs), so deleting it outright would remove real coverage.

**Day-one state (known, temporary).** On adoption, the platform inherits exactly one coherence violation: the placeholder `rusty_bucket` (`candidate`) delivers to the production-tier `fao` consumer — a pre-production shakedown. During the transition, the tier-rule check **warns, not blocks**, on this edge, until the real production ensemble is graduated. The migration is *not* gated on a hasty graduation.

## 12. Decided vs Deferred/Open

**Decided (this review, 2026-07-27):**

- Only `graduate` sources write to the prod shelf; **candidates go nowhere central, for now.**
- Three shelf-write guards: `-p` intent / `graduate` eligibility / credentials. **Keep `-p`.**
- Gate split by boundary: maturity gates the write, delivery gates the serve; the producer never reads consumer config.
- Maturity rules **R1** (no retired member in an active ensemble) + **R2** (graduate ensemble → all-graduate members).

**Deferred / open (stated plainly — not smuggled as decided):**

- **Delivery-unit home:** `postprocessors/` (exists) vs a new `deliveries/` folder vs both.
- **Delivery-unit weight:** is every delivery unit a full postprocessor (like `un_fao`), or can it be lightweight when no transformation is needed?
- **Scheduled candidate / shadow ensembles** (e.g. a `views_shadow` endpoint run monthly): shared prod shelf tier-tagged, or a separate shadow shelf? *Revisit soon.*
- **Two stores today** (the legacy `views-forecasts` store for the public API + the Appwrite `production_forecasts` shelf): eventual consolidation, and the maintainer's "on Hetzner one day" intent — not now.
- **`config_meta.py` cleanup:** the "documentation only" docstring must stop being a lie once delivery moves out.
- **The guard's stale-log gap:** even the live `deprecated`-member check reads the member's *artifact-time log*, so a member deprecated *after* its artifacts were generated slips through (register-worthy, Tier-3).

## References

- ADR-001 (ontology), ADR-002 (topology), ADR-003 (authority — the `deployment_status` allowed-list), ADR-016 (the same derive-don't-declare instinct).
- views-postprocessing **ADR-013** (the wire half; see §6).
- pipeline-core **Lean Platform End-State Roadmap** (`documentation/plans/2026-07-27_lean_platform_end_state_roadmap.md`) — owns the sequencing/retirement this ADR's §1 direction-of-travel defers; the basis for this ADR's acceptance.
- `reports/expert_reviews/2026-07-19_adr013_wire_contract_review_views_models_seat.md`.
- `tests/test_deployment_status_inert.py` (pins the "label is inert" fact from §2).
- `tools/liveness` / epic #238 (the instrument in §7).
- pipeline-core: `modules/validation/core_config_sniffer.py`; `modules/validation/ensemble/check.py` (the dead guard); `managers/prediction/io.py` (gen-1: one method, two stores, `type=target`); `managers/model/model.py:572` (the gen-2 conditional saver trio); `cli/args.py` (`-m`); **`tests/test_managers/test_delivery_characterization.py`** (pins the three producer generations + the two shelf dialects).
- Session design discussion, 2026-07-02; grounding revision, 2026-07-27.
