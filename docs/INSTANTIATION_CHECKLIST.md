# Instantiation Checklist

Completed during initial adoption of governance docs for views-models.

---

## ADR Adaptation

### All adopted ADRs
- [x] Update Status from template to Accepted
- [x] Fill in Date (2026-03-15), Deciders (Simon), Informed (All contributors)

### Per-ADR adaptation notes
- [x] **ADR-000:** Path set to `docs/ADRs/`
- [x] **ADR-001:** Ontology categories mapped to views-models entities (models, configs, ensembles, tooling, etc.)
- [x] **ADR-002:** Dependency direction defined (models → external packages, configs → common/)
- [x] **ADR-003:** Fail-loud examples adapted to config validation and partition consistency
- [x] **ADR-004:** Deferred (evolution/stability rules not yet needed)
- [x] **ADR-005:** Testing taxonomy mapped to existing test suite (green/beige categories)
- [x] **ADR-006:** Intent contracts identified for 4 entities
- [x] **ADR-007:** Silicon agent constraints adapted for bulk config operations
- [x] **ADR-008:** Observability grounded in current logging patterns and known deviations
- [x] **ADR-009:** Boundary contracts mapped to config validation tests

---

## CICs

- [x] Replace placeholder active contracts list in `CICs/README.md`
- [x] Create `ModelScaffoldBuilder.md`
- [x] Create `EnsembleScaffoldBuilder.md`
- [x] Create `CommonPartitions.md`
- [x] Create `CatalogExtractor.md`

---

## Contributor Protocols

- [x] Adapt `carbon_based_agents.md` for views-models team and conventions
- [x] Adapt `silicon_based_agents.md` for bulk config operations and TDD workflow
- [x] Adapt hardened protocol for ML forecasting domain (partition integrity, ensemble ordering)

---

## Standards

- [x] Adapt `logging_and_observability_standard.md` to current patterns and known deviations
- [ ] Physical architecture standard — skipped (not applicable to this repo's architecture)

---

## Final Verification

- [x] No files still have Status `--template--`
- [x] No phantom references to non-existent files
- [x] All cross-ADR references resolve correctly
- [ ] Run `validate_docs.sh` to check internal consistency
