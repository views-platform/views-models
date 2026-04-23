
# ADR-010: Technical Risk Register as a Governance Artifact

**Status:** Accepted  
**Date:** 2026-04-05  
**Deciders:** Project maintainers  
**Informed:** All contributors  

---

## Context

Repository assimilation (April 2026) identified 11 structural risks in views-models, ranging from partition coordination fragility (high severity) to untested scaffold builders (medium severity). These risks are architectural — they emerge from design decisions, not from bugs in any single file.

Without a persistent, structured register, risks are:
- Discovered during audits but forgotten between them
- Discussed informally but not tracked to resolution
- Rediscovered by new contributors who lack context on prior analysis

---

## Decision

The repository maintains a **Technical Risk Register** at `reports/technical_risk_register.md`.

### Concern Format

Each entry uses this format:

| Field | Description |
|---|---|
| **ID** | `C-xx` for concerns, `D-xx` for disagreements |
| **Tier** | 1 (critical) through 4 (informational) |
| **Title** | Short description |
| **Trigger** | The specific circumstance under which the risk becomes actionable |
| **Source** | How this concern was identified (e.g., repo-assimilation, expert review, incident) |
| **Status** | Open, Mitigated, Accepted, Resolved |
| **Notes** | Additional context, references to related ADRs or PRs |

### Tier Definitions

| Tier | Meaning | Response |
|---|---|---|
| **1** | Critical structural risk; failure would affect multiple models or external consumers | Must be addressed or explicitly accepted with rationale |
| **2** | Significant risk; failure would affect a class of models or a governance mechanism | Should be addressed in the next development cycle |
| **3** | Moderate risk; failure would cause inconvenience or require manual intervention | Address when adjacent work touches the area |
| **4** | Informational; noted for awareness | No action required unless promoted |

### When Entries Are Added

Concerns are opened during:
- Repository assimilation audits
- Expert code reviews
- Tech debt cleanup sessions
- Falsification audits
- Incident post-mortems

### When Entries Are Closed

Concerns are resolved when:
- The underlying risk is eliminated (code change + test)
- The risk is explicitly accepted with rationale (documented in Notes)
- The risk is superseded by a different concern

---

## Rationale

A structured register makes risks visible, trackable, and reviewable. It prevents the pattern of "we know about that problem" without any record of what "that problem" actually is.

---

## Consequences

### Positive
- Risks persist across conversations and contributors
- New contributors can quickly understand known architectural weaknesses
- Audit findings have a concrete landing place

### Negative
- Register requires maintenance; stale entries reduce trust
- Risk of "concern inflation" if trivial items are registered at high tiers

---

## References

- `reports/technical_risk_register.md` — the register itself
- [ADR-004](004_evolution.md) — Evolution rules that influence risk severity
- [ADR-005](005_testing.md) — Testing gaps are a common risk source
