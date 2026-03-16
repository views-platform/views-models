# ADR-006: Intent Contracts for Non-Trivial Classes

**Status:** Accepted
**Date:** 2026-03-15
**Deciders:** Simon (project maintainer)
**Informed:** All contributors

---

## Context

views-models contains several non-trivial classes and modules whose purpose, boundaries, and failure modes are not documented. `ModelScaffoldBuilder`, `EnsembleScaffoldBuilder`, and `common/partitions.generate()` all have implicit contracts that new contributors must reverse-engineer from code.

---

## Decision

Non-trivial classes and modules in this repository must have **Intent Contracts** — human-readable documents that declare purpose, responsibilities, boundaries, and failure semantics.

### When an Intent Contract is Required

An Intent Contract is mandatory for:
- Classes with lifecycle logic (build → configure → assess)
- Shared infrastructure modules that all models depend on
- Boundary functions that load external data or configs
- Orchestration components

### What an Intent Contract Must Define

1. Purpose (what the class/module is for)
2. Non-Goals (what it is explicitly NOT responsible for)
3. Responsibilities and Guarantees
4. Inputs and Assumptions
5. Outputs and Side Effects
6. Failure Modes
7. Boundaries and Interactions
8. Correct and Incorrect Usage Examples
9. Test Alignment
10. Known Deviations from ideal patterns

### Current Contracts

See `docs/CICs/README.md` for the active contracts list.

---

## Consequences

### Positive
- New contributors can understand component purpose without reading implementation
- Silicon-based agents (ADR-007) can reference contracts to verify changes
- Tests can be derived from contracts

### Negative
- Contracts must be maintained alongside code
- Contracts for simple config-returning functions may feel bureaucratic

---

## References

- `docs/CICs/` — Intent Contract directory
- ADR-003 (Authority of Declarations)
- ADR-005 (Testing)
