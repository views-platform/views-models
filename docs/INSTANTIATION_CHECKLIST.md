# Instantiation Checklist

Completed during initial adoption (2026-03-15) and updated (2026-04-05) with governance expansion.

---

## ADR Adaptation

### All adopted ADRs
- [x] Update Status from template to Accepted
- [x] Fill in Date, Deciders (Project maintainers), Informed (All contributors)

### Per-ADR adaptation notes
- [x] **ADR-000:** Path set to `docs/ADRs/`
- [x] **ADR-001:** Ontology categories mapped to views-models entities (68 models, 5 ensembles, configs, tooling, etc.)
- [x] **ADR-002:** Dependency direction defined (models → external packages, self-contained configs)
- [x] **ADR-003:** Fail-loud examples adapted to config validation and partition consistency
- [x] **ADR-004:** Activated (2026-04-05) — three-tier stability classification (Stable/Conventional/Volatile)
- [x] **ADR-005:** Testing taxonomy mapped to existing test suite (green/beige categories)
- [x] **ADR-006:** Intent contracts identified for 5 entities
- [x] **ADR-007:** Silicon agent constraints adapted for bulk config operations
- [x] **ADR-008:** Observability grounded in current logging patterns and known deviations
- [x] **ADR-009:** Boundary contracts mapped to config validation tests
- [x] **ADR-010:** Technical risk register ADR created, register seeded with 11 risks from repo-assimilation

---

## CICs

- [x] Replace placeholder active contracts list in `CICs/README.md`
- [x] Create `ModelScaffoldBuilder.md`
- [x] Create `EnsembleScaffoldBuilder.md`
- [x] Create `CatalogExtractor.md`
- [x] Create `PackageScaffoldBuilder.md` (2026-04-05)
- [x] Create `IntegrationTestRunner.md` (2026-04-05)

---

## Contributor Protocols

- [x] Adapt `carbon_based_agents.md` for views-models team and conventions
- [x] Adapt `silicon_based_agents.md` for bulk config operations and TDD workflow
- [x] Adapt hardened protocol for ML forecasting domain (partition integrity, ensemble ordering)

---

## Standards

- [x] Adapt `logging_and_observability_standard.md` to current patterns and known deviations
- [x] Adapt `physical_architecture_standard.md` as default heuristic (not strict enforcement) (2026-04-05)

---

## Risk Register

- [x] Create `reports/technical_risk_register.md` seeded with 11 concerns from repo-assimilation (2026-04-05)
- [x] Create governing ADR-010 (2026-04-05)

---

## Final Verification

- [x] No files still have Status `--template--`
- [x] No phantom references to non-existent files
- [x] All cross-ADR references resolve correctly
- [x] Run `validate_docs.sh` to check internal consistency
