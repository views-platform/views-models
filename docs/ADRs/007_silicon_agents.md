# ADR-007: Silicon-Based Agents as Untrusted Contributors

**Status:** Accepted
**Date:** 2026-03-15
**Deciders:** Simon (project maintainer)
**Informed:** All contributors

---

## Context

This repository is actively modified by AI-assisted tooling (Claude Code). Recent bulk changes include: migrating 13 model `main.py` files, adding config keys to 52+ models, and replacing 66 partition config files. These changes carry risks of silent semantic corruption — e.g., the `{{e}}` double-brace bug introduced during template-based CLI migration.

---

## Decision

Silicon-based agents (LLM assistants, code generators, refactoring tools) are treated as **untrusted contributors**. They may accelerate work but must not own intent.

### Allowed Operations

Silicon-based agents may:
- Perform scoped refactors within a single class or file
- Add or update tests that reflect declared intent
- Implement changes explicitly requested by a human contributor
- Make mechanical changes (formatting, renaming) with no semantic impact
- Perform bulk changes to config files when a clear template is provided

### Forbidden Operations

Silicon-based agents must not:
- Introduce or modify model semantics without updating intent contracts
- Infer model behavior from naming conventions or file structure
- Remove validation, fail-loud behavior, or tests
- Cross architectural boundaries (ADR-002)
- Perform full-file rewrites of existing files without reading first (anti-truncation rule)

### Anti-Truncation Rule

When modifying existing files, silicon-based agents must:
1. Read the file first
2. Apply targeted, minimal edits
3. Leave unrelated content untouched

Full-file rewrites via template generation (as done during CLI migration) must be followed by verification against a known-good reference file.

### Mandatory Artifacts

Every silicon-based agent-assisted change must include:
- A summary of what was changed
- Reference to relevant ADRs
- Explicit declaration of uncertainty
- Verification that tests pass after the change

---

## Consequences

### Positive
- Prevents silent semantic corruption in bulk changes
- Forces verification of AI-generated code
- Makes human reviewers aware of heightened scrutiny needs

### Negative
- Slows down bulk operations
- Requires discipline in AI-assisted workflows

---

## References

- `docs/contributor_protocols/silicon_based_agents.md`
- ADR-003 (Authority of Declarations)
- ADR-005 (Testing)
