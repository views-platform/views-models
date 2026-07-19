# ADR-017: Forecast Sources, Composition, and Delivery — separating what a model *is*, what it's *built from*, and *where it goes*

**Status:** Proposed
**Date:** 2026-07-02
**Deciders:** Simon (maintainer) — *pending review*
**Consulted:** platform contributors
**Informed:** All contributors

---

## Context

The repo models forecasting entities as **models** and **ensembles**, each carrying a `config_deployment.py` with a single `deployment_status` field constrained to `{shadow, deployed, baseline, deprecated}` (enforced by pipeline-core `modules/validation/core_config_sniffer.py`).

In practice this field does almost nothing and no longer means what it was meant to:

- **It controls nothing operationally.** `monthly_run.sh` never reads it; the prediction path never routes on it. The *only* things that act on it are: the integration-test runner skips `deprecated`, the (broken) ensemble guard, and the catalog displays it.
- **It has drifted from its origin.** It began as a **composition switch** in a *one-ensemble* world: `deployed` meant "member of THE production ensemble," `shadow` meant "a challenger — monitored, not in production." With one ensemble and one destination, one label sufficed and "where does it go?" wasn't a question.
- **The system outgrew that.** Composition is now explicit (`config_modelset.py`), and there are now **multiple ensembles delivered to multiple stores/APIs** (the main forecast line and the FAO delivery via `un_fao`). `deployed` lost its job and decayed into a vague *"eligible"* — which collides with what `shadow` already means.

Concrete symptoms:

- `deployed` on a standalone model is inert — a note of intent.
- The ensemble↔constituent guard (`views-pipeline-core .../modules/validation/ensemble/check.py`) is **dead**: it branches on `single_model_dp_status == "production"`, a value the sniffer forbids, so the rule never fires. A `deployed` ensemble can silently contain `shadow` constituents — and `white_mustang` already does (`lavender_haze`, `blank_space`).
- **Delivery routing is implicit and scattered.** The FAO edge lives on the *consumer* (`postprocessors/un_fao/configs/config_meta.py` → `"ensemble": "rusty_bucket"`); the main line is `monthly_run.sh` + the store + the API with no explicit declaration. Nothing says, in one place, "here is where forecasts leave the building."
- **Delivering a lone model requires wrapping it in a one-model ensemble — a hack.** (The tell that a shared abstraction is missing.)

## Decision

Model the domain as **three orthogonal responsibilities**, each owned by the layer it belongs to.

### 1. A first-class abstraction: **forecast source**
Anything that emits a forecast is a *source*. A **model** is a *leaf* source; an **ensemble** is a *composite* source that combines other sources (the Composite pattern). Delivery and readiness depend on the **source** abstraction, never on `Ensemble` specifically. (This is why the one-model-ensemble wrapper exists today — no shared interface — and why it disappears here.)

### 2. Three axes, each declared/owned in one place
- **Maturity** — intrinsic to a source: `candidate → graduate → retired`. **Declared** on the source in `config_deployment.py`, *identically for models and ensembles*. Replaces `deployment_status`. `baseline` is **not** a maturity — it is a *role*, already captured by the model's algorithm + the existing `regression_point_baselines` references.
- **Composition** — a property of *composite* sources only (the members). **Declared** on the ensemble in `config_modelset.py` (**already exists**, unchanged).
- **Delivery** — a *relationship*, not a property of any source: which source's forecast ships to which consumer. **Declared in the delivery layer** (a postprocessor / delivery unit), **never on the source**.

### 3. "Deployed" / "in production" is **derived, never declared**
> A source is *in production* ⟺ its maturity is `graduate` **and** a delivery ships it (directly, or via a composite that contains it) to a **production-tier** consumer.

Because nobody writes it, it cannot lie.

### 4. Consumers are defined once
A small central registry (`meta/consumers.py`) names the valid consumers and their tiers, so destination names can't drift and coherence rules have a tier to check against.

### 5. Coherence is enforced at the delivery boundary, fail-loud
- a delivery to a **production-tier** consumer requires its source to be `graduate`;
- a `graduate` **composite** requires its members to be `graduate` (the white_mustang check);
- a delivery's `source` and `consumer` must resolve (a real source; a consumer in the registry).

## Rationale (against the maintainer's principles)

- **SRP** — an ensemble changes only for *composition*; delivery changes only in the delivery layer; maturity only on the source. Today `Ensemble` is also "the deliverable" (two reasons to change).
- **LSP / ISP / DIP** — delivery depends on the narrow `forecast source` interface, so a lone model substitutes for an ensemble freely; the one-model-ensemble hack becomes *impossible*. High-level delivery stops depending on the concrete `Ensemble`.
- **OCP** — a new consumer is a new delivery edge; a new kind of source implements the interface; the status enum stops being an edit-point.
- **CRP** — combination and delivery are not reused together, so they should not be forced together (today "ensemble" forces them).
- **SDP / SAP** — `forecast source` is the stable abstraction everything depends *toward*; concrete producers and delivery routes are the volatile leaves depending inward.
- **ADP** — delivery depends on the source abstraction, not the reverse; putting destinations on the producer would create a producer→consumer back-edge (cycle risk).
- **Screaming architecture** — an explicit delivery layer makes "where forecasts go out" *visible* instead of implicit across a bash list + a postprocessor config + a dead label.

## Considered Alternatives

### A — Keep `deployment_status`, add a `destinations` field on the source
Makes a *producer* know its *consumers*: the ensemble would change whenever a downstream delivery relationship changes (SRP/DIP violation), and it re-creates the cross-file coherence split. **Rejected.**

### B — One centralized routing config (ensemble → consumer map)
Takes the routing decision out of the delivery layer, duplicates membership already encoded elsewhere, and becomes a merge-conflict bottleneck. **Rejected as the source of truth** — a *derived* topology view gives the same at-a-glance benefit without the drift.

### C — Derive routing from status alone (shadow→shadow bucket, deployed→prod bucket)
Status alone cannot express "deployed to FAO but not to the main API" — multiple production destinations exist. **Rejected as insufficient.**

## Consequences

### Positive
- `deployed` cannot lie (derived); the white_mustang class of incoherence is caught automatically at the delivery boundary.
- A lone model is directly deliverable — no wrapper.
- Delivery topology is explicit and *screamable*; the aggregate view is generated, not hand-kept.
- Each config file answers exactly one question.

### Negative / costs
- The `deployment_status → maturity` rename is a **cross-repo breaking contract change** (views-models configs + pipeline-core sniffer/templates/guard + ADR-003) against a **published** pipeline-core — the C-73 / C-132 skew tax.
- Making the **main line an explicit delivery unit** is new structure touching the most operationally sensitive path.
- A migration is required (phased below).

## Implementation Notes (phasing — deliberately not a big-bang)

**Distance-to-clarity ≪ distance-to-purity.** The value can be captured incrementally, mostly local to views-models, before paying the cross-repo/operational costs.

- **Phase 0 (this ADR):** the shared model. Zero code.
- **Phase 1 — cheap, local, breaks nothing published (≈80% of the clarity):**
  - fix the dead guard (`check.py` `"production" → "deployed"`);
  - make delivery first-class for FAO: `un_fao` config `ensemble → source` + an explicit `consumer`; add `meta/consumers.py`;
  - derive `is_in_production` and repoint the label reads.
- **Phase 2 — cross-repo, deliberate:** `deployment_status → maturity` + value remap across all sources + the pipeline-core contract + ADR-003.
- **Phase 3 — structural, operationally sensitive:** make the main line an explicit delivery unit (`deliveries/main_forecasts/`), so *all* delivery is uniform and explicit.
- **Phase 4:** flip readiness/coherence checks to the derived, delivery-boundary form.

### Target-state config shapes

A source declares only its maturity (models and ensembles, same shape):
```python
# models/<m>/configs/config_deployment.py  |  ensembles/<e>/configs/config_deployment.py
def get_deployment_config():
    return {"maturity": "candidate"}          # candidate | graduate | retired
```

An ensemble additionally declares its members (unchanged):
```python
# ensembles/<e>/configs/config_modelset.py
def get_modelset_config():
    return {"models": [...]}
```

A delivery unit declares the edge — the only place "goes where" lives:
```python
# postprocessors/un_fao/configs/config_delivery.py
def get_delivery_config():
    return {"source": "rusty_bucket", "consumer": "fao"}   # source = a model OR an ensemble
# deliveries/main_forecasts/configs/config_delivery.py
def get_delivery_config():
    return {"sources": ["pink_ponyclub", ...], "consumer": "main_api"}
```

Consumers defined once:
```python
# meta/consumers.py
CONSUMERS = {"main_api": {"tier": "prod"}, "fao": {"tier": "prod"}, "shadow_bucket": {"tier": "staging"}}
```

Derived, stored nowhere:
```python
is_in_production(source) == (maturity(source) == "graduate") and delivered_to_prod(source)
```

## Validation & Monitoring
- Coherence checks fail loud at config-load / pre-delivery: candidate→prod, non-graduate members in a graduate composite, unknown source/consumer.
- A generated "delivery topology" view (like the catalog) lists every `source → consumer` edge and every in-production source — replacing trust-the-label.

## Open Questions
- Final vocabulary for the maturity axis (`candidate / graduate / retired`?).
- Does a lone model ever deliver directly today? (Currently **no** — but the model must support it without the wrapper.)
- Where do delivery units physically live — under `postprocessors/`, a new `deliveries/`, or both?
- `baseline`: fully derived from references, or does a source ever need to declare "I am a baseline"?
- Migration for `un_fao`: producer-declares vs consumer-declares — validate-they-agree, or discover?

## References
- ADR-001 (ontology), ADR-002 (topology), ADR-003 (authority — the `deployment_status` allowed-list), ADR-016 (point/stochastic — the same "derive, don't declare" instinct).
- views-pipeline-core `modules/validation/core_config_sniffer.py` (`SUPPORTED_DEPLOYMENT_STATUSES`); `modules/validation/ensemble/check.py` (the dead `"production"` guard).
- Technical risk register — the deployment_status findings (ensemble↔constituent coherence gap; dead guard; implicit/emergent routing).
- Session design discussion, 2026-07-02.
