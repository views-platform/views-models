# ADR-018: One writer for the Appwrite environment, and setup lives in `bootstrap.sh`

**Status:** Accepted
**Date:** 2026-08-02
**Supersedes:** nothing. **Related:** ADR-002 (topology), ADR-003 (fail loud), ADR-017 (source/composition/delivery)

---

## Context

Two conventions were established by code in #308/#309/#311 and existed only as shell
comments and issue numbers. Both are the kind a contributor can violate without knowing
they exist, which is what this repo's CLAUDE.md says an ADR is for.

**The failure that produced them.** `postprocessors/un_fao/run.sh` had two blocks writing
the same environment variable names: `source .env` and a loop exporting values read from
the platform coordinate registry. The registry won because it ran second. Reversing the
two blocks would have inverted the semantics silently, with no test failing. Separately,
an unresolvable registry warned and continued, so the run died minutes later at the
datastore boundary describing a symptom rather than a cause.

**Why it kept recurring.** The reasoning lived in commit messages and closed issues. Three
successive changes to that file each re-derived part of it, and two of them shipped a
variant of the same defect (see Consequences).

---

## Decision

### 1. Coordinates come from the registry. The secret comes from the operator.

`tools/credentials/platform_env.sh` is the **only** writer of Appwrite coordinates and the
Appwrite secret. Consumers source it and call `platform_env_load`.

- Coordinates are read from the platform coordinate registry (homed in `views-appwrite`).
- The secret (`APPWRITE_DATASTORE_API_KEY`) comes from the process environment or `.env`.
- **A `.env` that declares a coordinate the registry owns is an error**, reported with the
  variable and both sources named. It is not resolved by precedence: precedence is what
  makes the outcome depend on line order.

### 2. An unresolvable or unreadable registry is fatal, and fatal early.

Checked before conda, before pip, before any work. Warning and continuing does not save a
run; it relocates the failure to a place that cannot explain it.

### 3. "Will the child see this?" is answered in exported scope only.

`[ -n "$VAR" ]` cannot distinguish an exported variable from a shell-local one, and every
consumer of this environment is a child process. Presence checks use
`platform_env_is_exported`, which reads `export -p`.

### 4. One-time machine setup belongs in `bootstrap.sh`.

`./bootstrap.sh` — no arguments, no companion document — is the entry point for a machine
that has never run this platform. It asks for **one secret and zero coordinates**. It is
idempotent and it is exercised in CI against a fixture registry with a fake secret, because
a setup path verified once on one laptop rots exactly like the prose it replaces.

It does **not** create conda environments; each `run.sh` still owns its own.

---

## Consequences

### Positive

- The order of two blocks in a shell script can no longer change what the platform reads.
- A machine-layout problem fails in the first second, naming the path and the override.
- Setup is executable, so it cannot be subtly wrong in the way a document can.

### Negative

- An existing `.env` carrying coordinates must have them deleted before the launcher will
  run. Provably a no-op — those values were never exported — but it is a manual step on
  every machine that has one.
- `bootstrap.sh` and each `run.sh` both source the shared file, so a change to the contract
  touches a file that production launchers depend on.
- The ~130 per-model `run.sh` scripts still carry their own copy of the macOS setup block.
  `bootstrap.sh` adds the canonical home; removing the copies is tracked separately
  (#310) because it touches 131 protected files.

### The rule this ADR exists to stop being rediscovered

**Twice in four days, a presence check tested shell scope while the consumer needed
exported scope**, and both times the code reported success while the child process
received nothing:

1. `_platform001_coordinate_state()` announced *"Coordinates ARE present in the environment
   (exported outside this script)"* about values `source .env` had set and nothing had
   exported.
2. `platform_env_export_secret`'s first draft skipped its own `export` because the guard saw
   the value that the launcher's earlier `source .env` had left in shell scope.

Both were written by an author who had just read the incident report for the first one.
That is the argument for this being an ADR rather than a comment. Tracked as **C-112**.

---

## References

- `tools/credentials/platform_env.sh` — the implementation
- `bootstrap.sh` — the entry point
- `.github/workflows/bootstrap.yml` — the CI proof
- `tests/test_platform_env.py` — behavioural tests of the contract
- Register: **C-112** (shell-vs-exported scope), C-47/C-48 (the originating findings)
- Issues: #308, #309, #311; the seam contract in `views-appwrite`
