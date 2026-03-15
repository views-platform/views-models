# ADR-000: Use of Architecture Decision Records (ADRs)

**Status:** Accepted
**Date:** 2026-03-15
**Deciders:** Simon (project maintainer)
**Informed:** All contributors

---

## Context

views-models is a monorepo containing ~66 forecasting models, 5 ensembles, data extractors, postprocessors, and tooling for the VIEWS conflict prediction platform. The repository has multiple contributors, evolving conventions, and a history of implicit decisions that have led to architectural drift (e.g., two CLI patterns, duplicated partition configs, inconsistent config keys).

Without a shared record of *why* decisions were made, the project risks:
- Re-litigating settled questions (e.g., why all models use the same partition boundaries)
- Accidental reversals of critical design choices
- Accumulating invisible technical debt
- Losing institutional memory as contributors change

---

## Decision

We will use **Architecture Decision Records (ADRs)** to document significant technical, architectural, and conceptual decisions in this project.

ADRs are:
- Written in Markdown
- Stored in the repository under `base_docs/ADRs/`
- Numbered sequentially
- Treated as first-class project artifacts

---

## When to Write an ADR

Write an ADR when making a decision that:
- Affects model configuration conventions or required config keys
- Changes partition boundaries, training windows, or evaluation methodology
- Introduces new shared infrastructure (e.g., `common/`)
- Changes the CLI API pattern or model launcher conventions
- Modifies ensemble reconciliation logic or CM/PGM ordering
- Affects the CI/CD pipeline or catalog generation

Do **not** write ADRs for:
- Adding a new model that follows existing conventions
- Routine hyperparameter changes within a single model
- Documentation-only changes

---

## Lifecycle

- **Proposed** — decision under consideration
- **Accepted** — decision is active and authoritative
- **Superseded** — replaced by a newer ADR
- **Deprecated** — decision remains but should no longer be used

Decisions are never deleted. If a decision changes, it is **superseded**, not erased.

---

## Consequences

### Positive
- Clearer decision-making across a multi-contributor forecasting platform
- Fewer repeated debates about config conventions
- Easier onboarding for new model developers
- Better long-term coherence as the model zoo grows

### Negative
- Small upfront cost in writing
- Requires discipline to maintain

---

## References

- `base_docs/ADRs/adr_template.md`
- `base_docs/ADRs/README.md`
