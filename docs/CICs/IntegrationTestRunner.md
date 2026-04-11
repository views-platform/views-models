
# Class Intent Contract: IntegrationTestRunner

**Status:** Active  
**Owner:** Project maintainers  
**Last reviewed:** 2026-04-11  
**Related ADRs:** ADR-004, ADR-005, ADR-008, ADR-009  

---

## 1. Purpose

`run_integration_tests.sh` is the only mechanism that tests actual model training and evaluation in views-models. It trains and evaluates each selected model on calibration and/or validation partitions using a shared conda environment, logs results per model, and produces a pass/fail summary. It never aborts on individual model failure — every model gets its turn.

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** run forecasting partitions (only calibration and validation)
- Does **not** validate model output quality or prediction accuracy — only that training and evaluation complete without error
- Does **not** replace the pytest structural test suite; the two are complementary
- Does **not** manage or create conda environments — it activates an existing one
- Does **not** modify any model code, configs, or artifacts
- Does **not** run in CI — it is a local, manual tool

---

## 3. Responsibilities and Guarantees

- Guarantees that every matched model is executed for every requested partition, regardless of prior failures (no early abort *during the run phase*; classification errors during `--level` filtering are surfaced before the run phase begins, see exit code 2 below)
- Guarantees crash isolation: each model runs in its own subshell (`bash -c "..."`)
- Guarantees per-model timeout enforcement via `timeout` command (default: 1800 seconds)
- Guarantees that results are classified as exactly one of: `PASS`, `FAIL(exit_code)`, `TIMEOUT`
- Guarantees that per-model stdout/stderr is captured to `$LOG_DIR/$partition/$model.log`
- Guarantees a structured summary log at `$LOG_DIR/summary.log`
- Guarantees exit code 0 only if all runs pass; exit code 1 if any fail or timeout; exit code 2 if any model in the candidate set cannot be classified by the `--level` filter (fail-fast before any model runs)

---

## 4. Inputs and Assumptions

### CLI Arguments

| Flag | Default | Description |
|---|---|---|
| `--env` | `views_pipeline` | Conda environment name to activate |
| `--models` | (all) | Space-separated model names to include |
| `--level` | (all) | Filter by level: `cm` or `pgm` |
| `--library` | (all) | Filter by algorithm library: `baseline`, `stepshifter`, `r2darts2`, `hydranet` |
| `--exclude` | `purple_alien` | Space-separated model names to skip |
| `--partitions` | `calibration validation` | Space-separated partition names |
| `--timeout` | `1800` | Seconds per model per partition |

### Assumptions

- The named conda environment exists and has all model dependencies installed
- Each model directory has `main.py` accepting `-r <partition> -t -e` arguments
- `--level` filtering requires `config_meta.py` to be loadable by Python 3 with `importlib`
- `--library` filtering requires `requirements.txt` to contain the library name (e.g., `views-baseline`)
- Model discovery uses `main.py` existence as the inclusion criterion

---

## 5. Outputs and Side Effects

- **Log directory:** `logs/integration_test_$TIMESTAMP/` with per-partition subdirectories
- **Per-model logs:** `$partition/$model.log` — complete stdout/stderr from training + evaluation
- **Summary log:** `summary.log` — tabular pass/fail results
- **Console output:** Colorized progress and summary table
- **Exit code:** 0 (all pass) or 1 (any fail/timeout)
- **Side effects on model directories:** Training produces artifacts in `models/$model/artifacts/` and W&B logs in `models/$model/wandb/` — these are normal training side effects, not introduced by the test runner

---

## 6. Failure Modes and Loudness

| Condition | Behavior |
|---|---|
| Model training crashes | Captured in log; classified as `FAIL(exit_code)`; script continues |
| Model exceeds timeout | Killed by `timeout`; classified as `TIMEOUT`; script continues |
| No models match filters | Prints "No models found to test"; exits 1 |
| Conda environment doesn't exist | Activation fails inside subshell; model classified as `FAIL` |
| `config_meta.py` unloadable (during `--level` filter) | Python stderr captured; error printed to stderr with model name + last traceback line; model collected in `CLASSIFICATION_ERRORS`; script exits 2 before running any models |
| Unknown CLI flag | Prints error; exits 1 |

The runner itself never fails silently. Individual model failures are captured and reported, not swallowed.

---

## 7. Boundaries and Interactions

| Interacts With | Direction | Nature |
|---|---|---|
| `models/*/main.py` | Invokes | Subprocess via `python main.py -r $partition -t -e` |
| `models/*/configs/config_meta.py` | Reads (for `--level` filter) | `importlib.util` from Python |
| `models/*/requirements.txt` | Reads (for `--library` filter) | `grep` for package name |
| Conda | Activates | `conda activate $ENV` in subshell |
| `logs/` | Writes | Timestamped log directories |

Does **not** interact with: ensemble directories, APIs, extractors, postprocessors, or the pytest test suite.

---

## 8. Examples of Correct Usage

```bash
# Run all models on both partitions
bash run_integration_tests.sh

# Run only country-month stepshifter models on calibration
bash run_integration_tests.sh --level cm --library stepshifter --partitions "calibration"

# Run specific models with 60-minute timeout
bash run_integration_tests.sh --models "counting_stars bad_blood" --timeout 3600

# Use a different conda environment
bash run_integration_tests.sh --env views_r2darts2
```

---

## 9. Examples of Incorrect Usage

```bash
# Wrong: expecting this to run forecasting
bash run_integration_tests.sh --partitions "forecasting"
# main.py will attempt to forecast, but the runner is designed for
# calibration/validation where ground truth exists

# Wrong: assuming --exclude appends to defaults
bash run_integration_tests.sh --exclude "new_model"
# This REPLACES the default exclusion (purple_alien), not appends to it

# Wrong: expecting this to run in CI
# The runner takes hours and requires a GPU-capable environment;
# it is not wired into .github/workflows/
```

---

## 10. Test Alignment

- **No tests exist** for the integration test runner itself.
- The runner IS the test mechanism for behavioral (green-team) model validation.
- ADR-005 documents it as complementary to the pytest structural suite.
- Known gap: not in CI (Risk Register C-03).

---

## 11. Evolution Notes

- If integrated into CI (even partially — e.g., a subset of models), the timeout defaults and filtering logic would need revisiting.
- The `--level` filter shells out to Python to load `config_meta.py` — this is fragile if `config_meta.py` gains new dependencies.
- The `--exclude` flag replaces rather than appends; this is a common source of user surprise and a candidate for change.

---

## 12. Known Deviations

- **Not in CI:** The only behavioral test mechanism is manual (Risk Register C-03). A model can be merged broken.
- **`--exclude` replaces defaults:** Documented in `--help` but surprising — passing `--exclude "foo"` removes the default `purple_alien` exclusion.
- **No ensemble coverage:** The runner only discovers models in `models/`; ensembles in `ensembles/` are not tested by this mechanism.
- **`--library` filter silently excludes models lacking `requirements.txt`:** A model without a `requirements.txt` cannot be classified by the `--library` filter and is silently dropped from the filtered set. Tracked as Risk Register C-34.

---

## End of Contract

This document defines the **intended meaning** of `run_integration_tests.sh`.

Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
