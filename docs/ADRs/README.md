# ADR README and Governance Map

This repository uses Architectural Decision Records (ADRs) to govern
structural, semantic, and operational behavior.

---

## Constitutional ADRs (000-009)

These ADRs define system philosophy and governance:

- **[ADR-000](000_use_of_adrs.md)** — Use of Architecture Decision Records
- **[ADR-001](001_ontology.md)** — Ontology of the Repository
- **[ADR-002](002_topology.md)** — Topology and Dependency Rules
- **[ADR-003](003_authority.md)** — Authority of Declarations Over Inference
- **[ADR-004](004_evolution.md)** — Rules for Evolution and Stability
- **[ADR-005](005_testing.md)** — Testing as Mandatory Critical Infrastructure
- **[ADR-006](006_intent_contracts.md)** — Intent Contracts for Non-Trivial Classes
- **[ADR-007](007_silicon_agents.md)** — Silicon-Based Agents as Untrusted Contributors
- **[ADR-008](008_observability.md)** — Observability and Explicit Failure
- **[ADR-009](009_boundary_contracts.md)** — Boundary Contracts and Configuration Validation

---

## Governance Structure

- **Ontology (001)** defines what exists.
- **Topology (002)** defines structural direction.
- **Authority (003)** defines who owns meaning.
- **Evolution (004)** defines stability tiers and change rules.
- **Boundary Contracts (009)** define interaction rules.
- **Observability (008)** enforces failure semantics.
- **Testing (005)** verifies system integrity.
- **Intent Contracts (006)** bind class-level behavior.
- **Automation Governance (007)** constrains silicon-based agents.
- **Risk Register (010)** tracks structural risks.

---

## Project-Specific ADRs (010+)

- **[ADR-010](010_technical_risk_register.md)** — Technical Risk Register as a Governance Artifact
- **[ADR-011](011_partition_semantics.md)** — Partition Boundary Semantics
- **[ADR-012](012_target_scale_and_prefix_convention.md)** — Target Scale and Prefix Convention
- **[ADR-013](013_regression_target_name_agnosticism.md)** — Regression-Target Name Agnosticism (config is the single source of truth)
- **[ADR-014](014_reconciliation_composition_root.md)** — Reconciliation Composition Root (the sanctioned DIP wiring layer for the reconciler port)
- **[ADR-015](015_posterior_sample_count_standard.md)** — Posterior Sample-Count Standard and the Ensemble Constituent Contract
- **[ADR-016](016_point_stochastic_readiness.md)** — Point/Stochastic Discriminator for PredictionFrame Readiness
- **[vmo_017](017_source_composition_delivery.md)** — Forecast Sources, Composition, and Delivery (the three axes)
- **[ADR-018](018_environment_single_writer.md)** — One writer for the Appwrite environment; setup lives in `bootstrap.sh`
- **[ADR-019](019_delivery_declaration.md)** — The delivery declaration — one file per consumer
- **[ADR-020](020_errors_must_descend.md)** — Errors must descend, and must say where the stairs end
- **[ADR-021](021_coverage_is_declared_once.md)** — Coverage is declared once, in the delivery
- **[ADR-022](022_the_launcher_body_has_one_home.md)** — The delivery-protocol body has one home; a partner launcher is a wrapper

### Why one of these carries a `vmo_` prefix

**`vmo_017` is disambiguated because the number collides across three repositories** (#393):

| repo | prefix | its ADR-017 |
|---|---|---|
| views-models | `vmo_` | *Forecast Sources, Composition, and Delivery* |
| views-postprocessing | `vpp_` | *Facts shared with a repository we cannot read* |
| views-crafdapi | `vcr_` | *Reference Data in Repository* |

A bare "ADR-017" in a cross-repo sentence resolves to the **wrong document** for a reader
sitting in a repo that has its own 017 — and that is not hypothetical: views-crafdapi's
ADR-033 qualifies the citation once and then drops the qualifier four times in the same
passage, where every occurrence means *this* repo's 017.

**The number does not change and no existing citation breaks** — the prefix is additive.
Intra-repo prose may stay bare; write `vmo_017` wherever the sentence is read from, or
could be read from, another repository. The convention is not yet platform-wide; it is
being applied where a live collision forced it (views-postprocessing#264,
views-crafdapi#58).

Candidates for future ADRs:

- CM-before-PGM Ensemble Ordering
- Config Key Evolution Policy (how to add new required keys)
- Model Naming Convention and Governance
- Conda Environment Sharing via run.sh

---

## Recommended Adoption Order

### Phase 1 — Foundation (Done)
- ADR-000, ADR-003, ADR-008

### Phase 2 — Structure (Done)
- ADR-001, ADR-002

### Phase 3 — Testing & Intent (Done)
- ADR-005, ADR-006

### Phase 4 — Boundaries & Automation (Done)
- ADR-007, ADR-009
