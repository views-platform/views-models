
# Carbon-Based Agent Protocol
*(For contributors composed primarily of carbon, caffeine, and responsibility)*

**Status:** Active
**Applies to:** All human contributors
**Authority:** ADR-000 through ADR-009

---

## Purpose

This protocol defines the responsibilities, expectations, and obligations
of **carbon-based agents** contributing to the views-models repository.

Carbon-based agents are entrusted with:
- intent,
- judgment,
- and architectural authority.

---

## Core Principle: Stewardship of Intent

Carbon-based agents are **stewards of intent**, not merely authors of code.

In views-models, stewardship means:
- preserving config conventions across ~66 models,
- enforcing partition boundary consistency,
- preventing silent divergence of model configurations,
- and ensuring new models follow established patterns.

**Intent must not drift silently.**

---

## Ownership of Intent and Semantics

Carbon-based agents:
- own system intent and meaning,
- declare semantics explicitly in config files (ADR-003),
- and are accountable for their correctness.

If a change alters config conventions:
- the relevant tests must be updated,
- an ADR must be written for significant changes, or
- the change must not be merged.

---

## Fail-Loud Is a Moral Obligation

Silent failure is unacceptable.

Introducing:
- implicit defaults for missing config keys,
- fallback logic that hides config errors,
- or partition boundaries that differ from the canonical values enforced by tests

is considered a defect, even if the model runs successfully.

---

## Testing Is Part of the Change

A change is incomplete if it:
- cannot be validated by `pytest tests/`,
- weakens existing config validation tests,
- or introduces a new convention without a corresponding test.

When adding a new required config key:
1. Add it to `REQUIRED_META_KEYS` or `REQUIRED_HP_KEYS` in `tests/test_config_completeness.py`
2. Add the key to all existing models
3. Verify with `pytest tests/ -v`

---

## Interaction with Silicon-Based Agents

Using silicon-based agents does **not** reduce responsibility.

When carbon-based agents use silicon-based agents for bulk changes, they must:
- verify the changes against a known-good reference,
- run the full test suite,
- check for template artifacts (e.g., `{{e}}` double-brace bugs),
- and take full responsibility for the result.

---

## Non-Negotiable Expectations

Carbon-based agents must not:
- merge changes they do not understand,
- add models that skip required config keys,
- bypass tests under time pressure,
- introduce model-specific partition boundaries, or
- use the old CLI pattern (`parse_args` + `wandb.login()`).

---

## Final Note

Carbon-based agents are the **last line of defense**.

This protocol exists to ensure that,
even under pressure,
**the model zoo continues to mean what we think it means**.
