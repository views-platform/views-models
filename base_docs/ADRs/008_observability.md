# ADR-008: Observability and Explicit Failure

**Status:** Accepted
**Date:** 2026-03-15
**Deciders:** Simon (project maintainer)
**Informed:** All contributors

---

## Context

views-models delegates all ML logic to external packages, but the thin launcher layer (`main.py`, config files, tooling scripts) still has failure modes that must be handled explicitly. Silent failures in config loading, partition computation, or catalog generation can propagate undetected.

---

## Decision

### Fail-Loud Invariant

Structural failures must be raised explicitly. Silent degradation is prohibited.

Specifically:
- `main.py` files must wrap `ModelPathManager` initialization in try/except and raise `RuntimeError` with context
- Config loading must use `importlib.util` (not `exec()`), which naturally raises on syntax errors
- Missing required config keys must be caught by tests (ADR-005), not silently defaulted
- Catalog generation (`create_catalogs.py`) should log errors per-model rather than crashing entirely

### Logging Standards

This repository follows the logging standard defined in `base_docs/standards/logging_and_observability_standard.md`.

Key rules:
- `ERROR` and `CRITICAL` must be both logged and raised
- Warnings must not mask invariant violations
- `print()` is not acceptable for structural diagnostics (use `logging`)

### Current State (Known Deviations)

- `create_catalogs.py` does not handle per-model errors — one broken config crashes the entire run
- Some models use `warnings.filterwarnings("ignore")` broadly, which could mask important warnings
- WandB experiment tracking provides some observability but is external to this repo
- No alerting is configured for CI failures

---

## Consequences

### Positive
- Config loading failures are visible at the point of error
- Test failures surface config drift before merge

### Negative
- Broad `warnings.filterwarnings("ignore")` suppresses ML library warnings that could be informative
- No per-model resilience in catalog generation yet

---

## References

- `base_docs/standards/logging_and_observability_standard.md`
- ADR-003 (Authority of Declarations)
- ADR-005 (Testing)
