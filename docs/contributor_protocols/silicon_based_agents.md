
# Silicon-Based Agent Protocol
*(For contributors composed primarily of silicon, statistics, and confidence)*

**Status:** Active
**Applies to:** All automated or AI-assisted code modification
**Authority:** ADR-007 (Silicon-Based Agents as Untrusted Contributors)

---

## Purpose

This document defines **mandatory operational constraints** under which
**silicon-based agents** (e.g., Claude Code, LLM assistants, code generators)
may interact with the views-models repository.

This protocol exists to prevent:
- silent config corruption across 66+ models,
- architectural erosion of shared conventions,
- template artifacts (double-braces, escaped characters),
- and hard-to-detect partial failures in bulk operations.

---

## Threat Model

Silicon-based agents are assumed to:
- optimize for local plausibility, not global correctness,
- produce template artifacts when generating code via f-strings,
- silently truncate files during full-file rewrites,
- collapse abstractions for convenience,
- and produce outputs that *look valid* while being semantically incomplete.

Silicon-based agents are therefore treated as **untrusted contributors**.

---

## Allowed Operations

Silicon-based agents **may**:
- Add required config keys to models (bulk operations with verification)
- Migrate `main.py` files to new CLI patterns
- Replace partition configs with shared module imports
- Add or update tests that reflect declared intent
- Create new documentation (ADRs, CICs)
- Perform scoped refactors within a single file

All allowed operations remain subject to carbon-based agent review.

---

## Forbidden Operations

Silicon-based agents **must not**:
- Change partition boundaries without explicit authorization
- Modify model hyperparameters or querysets
- Infer model algorithm from directory names or import statements
- Remove config validation tests or weaken test assertions
- Introduce `exec()` or `eval()` for config loading
- Cross model boundaries (modifying model A based on model B)
- Perform full-file rewrites without reading first

If a silicon-based agent cannot proceed without guessing, it must stop.

---

## Mandatory Safety: Anti-Truncation Rule

When modifying existing files:
1. **Read the file first** using the Read tool
2. **Apply targeted edits** to specific locations
3. **Never use full-file Write on existing files** unless explicitly confirmed

When generating files from templates:
1. **Verify against a reference file** after generation
2. **Check for template artifacts** (double-braces `{{`, escaped characters)
3. **Run tests** to verify correctness

---

## Mandatory Safety: Bulk Operations

When modifying multiple models (bulk changes):
1. Write and run tests FIRST (TDD approach, ADR-005)
2. Apply changes via a script, not manual file-by-file edits
3. Verify with `pytest tests/ -v` after all changes
4. Spot-check 2-3 files manually against a known-good reference

---

## Required Artifacts

Every silicon-based agent-assisted change must include:
- Summary of what was changed and how many files were affected
- Reference to relevant ADRs
- Test results showing all tests pass
- Explicit note of any template-generated code that needs verification

---

## Enforcement

- Violations are treated as violations by the carbon-based agent who approved them
- Changes may be blocked solely on protocol grounds
- `pytest tests/` is the minimum verification bar

---

## Final Note

Silicon-based agents are tools, not collaborators.

This protocol exists to ensure that
**automation never outruns understanding**.
