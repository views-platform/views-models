# ADR-022: The delivery-protocol body has one home; a partner launcher is a wrapper

**Status:** Accepted
**Date:** 2026-08-11
**Deciders:** Simon, VIEWS platform team
**Related ADRs:** [ADR-018](018_environment_single_writer.md) (one writer for the Appwrite environment — the same shape, one layer down), [ADR-019](019_delivery_declaration.md) (the delivery declaration), [ADR-021](021_coverage_is_declared_once.md) (coverage declared once, derived everywhere), [ADR-020](020_errors_must_descend.md) (errors name the next file down)

---

## Context

Until 2026-08-11 there was one postprocessor, `un_fao`, and its `run.sh` was 136 lines:
the macOS block, the registry guard, the conda lifecycle, the pip install, the #294
capability assertion, the environment load, and the invocation of `main.py`.

Adding `un_crafd` (#333) meant deciding what to do with those 136 lines. Cloning them is
the obvious move and the wrong one, because **`run.sh` changes for two unrelated reasons**:

- because a **partner** is different — which conda environment, which views-postprocessing
  pin;
- because the **delivery protocol** is different — the registry check must precede conda,
  the environment must load after it (the registry parse needs 3.11 and the box's base is
  3.10), the capability assertion must read `config_meta` by import and not by grep.

The first genuinely varies per launcher. The second must not. Copying it means a protocol
fix has to be hand-applied once per partner, and the first one missed fails silently.

**That is not hypothetical, and the evidence is one repository away.** views-postprocessing
cloned `unfao/` into `crafd/` and then had to ship PR #211's follow-up commit, whose
message is *"every partner-scoped guard was scoped to ONE partner"*. They recorded the
extraction trigger as their C-33. views-crafdapi's #43 asked this repository not to repeat
it, by name.

There is a second, quieter cost. Every one of those 136 lines is a scar: #308 (registry
fatal and early), #293 (`set -a` truncates unquoted values at the first space), #294 (a
moving pin once carried no wire modules and would have shipped the legacy artifact green),
C-57 (a regex cannot tell a commented key from a live one), C-112 (a presence check in
shell scope answers a question about exported scope). A clone duplicates the lines and not
the understanding — the copy reads as boilerplate, and boilerplate gets tidied.

---

## Decision

### 1. The delivery protocol lives in `tools/launcher/postprocessor.sh`.

Sourced, never executed — the same shape as `tools/credentials/platform_env.sh`, and
non-executable for the same reason: an executable bit would claim an entry point the file
does not have. It defines one function, `postprocessor_launch`.

### 2. A partner launcher supplies variables and calls it.

`postprocessors/<consumer>/run.sh` declares only what varies by partner and delegates:

```bash
POSTPROCESSOR_ENV_NAME="views-postprocessing"
VIEWS_POSTPROCESSING_PIN="<git ref>"
script_path=$(dirname "$(realpath "$0")")
. "$( cd "$script_path/../../" >/dev/null 2>&1 && pwd )/tools/launcher/postprocessor.sh"
postprocessor_launch "$@"
```

`un_fao`'s went from 136 lines to 21.

### 3. What stays per-partner, and why that is not inconsistent.

| part | varies by | treatment |
|---|---|---|
| `configs/*.py` | partner | a real copy — each declares a different delivery |
| `main.py` (~28 lines) | partner | a copy; it names a different manager class and is too small to abstract |
| `run.sh` body | **the delivery protocol** | shared |

`configs/` looks like duplication and is not: the *values* differ per partner, and the ones
that must agree across the platform are already derived from `deliveries/<consumer>.py`
(ADR-019, ADR-021). What remains in a config is genuinely that partner's.

### 4. The guarantees are asserted for every launcher, not for the one we remembered.

`tests/test_postprocessor_launcher_environment.py` and
`tests/test_postprocessor_launcher_capability.py` are parametrised over every directory
under `postprocessors/` with a `run.sh`, and each reads the launcher's **effective** text —
its own file plus the body it sources. Reading `run.sh` alone would make every assertion
pass vacuously the moment the body moved: a green test measuring the wrong file.

Both files also assert that a launcher actually calls `postprocessor_launch`. A launcher
that grew its own copy of the protocol fails, which is the rule above expressed as a test
rather than as an intention.

---

## Consequences

### Positive

- A protocol fix is applied once and every partner gets it. That is the whole point.
- The scars live in one place, with their issue numbers, where a reader meets them once.
- A third partner costs a wrapper, not a review of 136 lines.
- The guarantees generalised for free: assertions that covered FAO now cover CRAF'd, and
  will cover whoever is next.

### Negative

- **The shared body is sourced by production launchers, so a change to it touches every
  delivery at once.** That is the cost of removing the duplication, not an argument against
  it — but it means the file deserves the same care as `platform_env.sh`, and a change to
  it is a change to the FAO delivery whether or not FAO is mentioned in the diff.
- A launcher is no longer readable top-to-bottom in one file. Mitigated by the wrapper
  naming the shared file and ADR, and by the body's header listing the variables a caller
  supplies.

### Deliberately not done

- **Extracting `configs/` or `main.py`.** Two partners is the trigger to share the thing
  that must not vary; it is not a licence to abstract the things that must. WET stays WET
  where duplication is the honest description.
- **Unifying the pins.** `un_fao` still installs `@main`, a moving pointer; `un_crafd` pins
  an immutable commit. That difference is real and is #364's to resolve, not a side effect
  of this extraction. Parameterising the pin is what makes #364 a one-variable edit.

### The extraction trigger, restated for the next partner

This is n=2 and the shared body is already justified. **The next thing to extract is
whatever the third partner shows you** — and the signal to watch for is the one this ADR
was written from: *a fix hand-applied in two places, where one of them was missed.*

---

## References

- `tools/launcher/postprocessor.sh` — the body; `tools/launcher/README.md` — its stated purpose
- `postprocessors/un_fao/run.sh`, `postprocessors/un_crafd/run.sh` — the wrappers
- `tests/test_postprocessor_launcher_{environment,capability}.py` — the guarantees, per launcher
- views-postprocessing #211 and their C-33 — the same clone, one repo away, and its scar
- views-crafdapi #43 — the request not to repeat it
- Register: **C-134** (the launcher clone and its extraction trigger), C-57, C-112, and
  issues #293, #294, #308, #309 — the scars the body carries
