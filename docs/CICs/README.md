# Class Intent Contracts README

This directory contains **Intent Contracts** as defined in ADR-006.

An Intent Contract is a human-readable, unambiguous declaration of:

- what a non-trivial class or module is meant to do,
- what it must never do,
- its invariants,
- and its failure semantics.

---

## Active Contracts

- `ModelScaffoldBuilder.md`
- `EnsembleScaffoldBuilder.md`
- `CatalogExtractor.md`

---

## Governance Relationship

Intent Contracts are governed by:

- ADR-006 (Intent Contracts for Non-Trivial Classes)
- ADR-003 (Authority of Declarations)
- ADR-005 (Testing Doctrine)

If a class changes meaning, its Intent Contract must be updated.
