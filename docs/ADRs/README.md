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

Candidates for future ADRs:

- Partition Boundary Semantics — ViewsMonth 121 = Jan 1990, 444 = Dec 2016, 492 = Dec 2020, 540 = Dec 2024; rationale for these split points is undocumented (C-21)
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
