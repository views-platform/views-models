# Integration Test Runner Guide

`run_integration_tests.sh` is a regression safety net: it verifies that changes in this repo or in upstream/downstream packages haven't broken model training or evaluation. It trains and evaluates every model in `models/` on the calibration and validation partitions, running them **sequentially** using a **single shared conda environment**, logging each result independently, and **never aborting on failure** — every model gets its turn regardless of what happened before it.

Each model run executes:

```
python main.py -r <partition> -t -e
```

That is: train (`-t`) and evaluate (`-e`) against the given partition (`-r`).


## Prerequisites

- **Conda** must be available on `PATH` (the script calls `conda shell.bash hook` internally).
- The target conda environment must already exist and have all model dependencies installed. The script does **not** create or install into environments — it only activates one.
- The default environment is `views_pipeline`. If your models require different packages than what's in that env, use `--env` to point at a different one.


## Quick Start

```bash
# Run all models (calibration + validation), default env, default timeout
bash run_integration_tests.sh

# Run two specific models only
bash run_integration_tests.sh --models "counting_stars bad_blood"

# Run only country-month models on calibration only
bash run_integration_tests.sh --level cm --partitions "calibration"

# Run only baseline models
bash run_integration_tests.sh --library baseline
```


## Options

| Flag | Value | Default | Description |
|------|-------|---------|-------------|
| `--models` | `"name1 name2 ..."` | *(all models)* | Run **only** these models. Names are space-separated inside quotes. Each name must match a directory under `models/` that contains a `main.py`. Names not found are skipped with a warning. |
| `--level` | `cm` or `pgm` | *(no filter)* | Run only models whose `config_meta.py` reports this level of analysis. The script reads each model's config via Python to check. Models whose level cannot be read are silently excluded. |
| `--library` | `baseline`, `stepshifter`, `r2darts2`, or `hydranet` | *(no filter)* | Run only models that depend on this architecture library. Determined by matching `views-<name>` in each model's `requirements.txt`. Can be combined with `--level`. |
| `--exclude` | `"name1 name2 ..."` | `"purple_alien"` | Skip these models. **Replaces** the default exclusion list — it does not append to it. To exclude nothing, pass an empty string: `--exclude ""`. |
| `--partitions` | `"p1 p2 ..."` | `"calibration validation"` | Which partitions to test. Valid values are `calibration`, `validation`, and `forecasting`. Space-separated inside quotes. |
| `--timeout` | seconds | `1800` (30 min) | Maximum wall-clock time per individual model run (one model x one partition). If exceeded, the run is killed and recorded as `TIMEOUT`. |
| `--env` | name | `views_pipeline` | Conda environment to activate before each model run. Can be an environment name or a path to a prefix. |
| `--help`, `-h` | — | — | Print usage summary and exit. |


## How It Works

### 1. Model Discovery

If `--models` is provided, the script uses that explicit list. Otherwise it scans every subdirectory of `models/` for a `main.py` file and sorts the results alphabetically.

Either way, models in the exclusion list are removed before anything runs.

### 2. Level Filtering (optional)

When `--level` is set, the script shells out to Python for each discovered model:

```python
import importlib.util
spec = importlib.util.spec_from_file_location('m', '<model>/configs/config_meta.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print(mod.get_meta_config().get('level', ''))
```

Only models whose `level` matches the filter survive. If a model's config can't be loaded, it is silently dropped.

### 3. Library Filtering (optional)

When `--library` is set, the script checks each model's `requirements.txt` for a line containing `views-<library>`. For example, `--library baseline` keeps only models whose `requirements.txt` contains `views-baseline`. This is a pure text match — no Python needed.

`--library` and `--level` can be combined to narrow further (e.g., `--library stepshifter --level cm` runs only country-month stepshifter models).

### 4. Execution

Before the run loop, the script classifies each model by `deployment_status` (loaded from `configs/config_deployment.py`). Models with `deployment_status == "deprecated"` are skipped — they are expected to fail by design, and running them would clutter the `FAIL` column. They appear in the summary as `DEPRECATED` instead. If `config_deployment.py` fails to load for any model, the script fails fast with exit code 2 before running anything (same behavior as a broken `config_meta.py` under `--level` filtering).

For each runnable model, for each partition, the script runs:

```bash
timeout --foreground $TIMEOUT bash -c "
    eval \"\$(conda shell.bash hook)\"
    conda activate '$CONDA_ENV'
    cd '$MODELS_DIR/$model'
    python main.py -r '$partition' -t -e
" > "$model_log" 2>&1
```

Key points:
- Each run gets its **own subshell** — a model crash cannot kill the runner.
- **All stdout and stderr** are captured to a log file, not printed to the terminal.
- The terminal shows only: `[N/TOTAL] model_name (partition)... PASS/FAIL/TIMEOUT/ABORTED (duration)`.
- `timeout --foreground` keeps the child process tree in the script's process group so that terminal `Ctrl-C` reaches the running model directly (see §Cancelling a run).

### 5. Result Classification

| Exit Code | Result | Meaning |
|-----------|--------|---------|
| `0` | `PASS` | Model trained and evaluated successfully. |
| `124` | `TIMEOUT` | Model exceeded the per-run timeout and was killed. |
| `130` | `ABORTED` | User pressed `Ctrl-C`; the current run was killed and remaining runs skipped. |
| n/a | `DEPRECATED` | Model's `deployment_status` is `deprecated`; no run attempted. |
| anything else | `FAIL(code)` | Model crashed. The exit code is recorded. |

### 5a. Cancelling a run

The script traps `SIGINT` (`Ctrl-C`). A single `Ctrl-C` will:

1. Kill the currently-running model (python dies via the shared process group).
2. Label its slot in the summary as `ABORTED`.
3. Skip all remaining models and partitions.
4. Print the partial summary table showing what had run.
5. Exit with code `130` (standard `Ctrl-C` exit code).

You do **not** need to hit `Ctrl-C` repeatedly. A single press is enough. This relies on `timeout --foreground` keeping the child tree in the script's process group; without that flag, `Ctrl-C` would reach only the parent script and the child would run to natural completion.

### 6. Summary Output

After all runs complete, the script prints a colored table to the terminal:

```
Model                         calibration    validation
-----                         ----------     ----------
bad_blood                     PASS           PASS
bouncy_organ                  FAIL(1)        PASS
counting_stars                PASS           TIMEOUT
electric_relaxation           DEPRECATED     DEPRECATED
invisible_string              ABORTED        SKIPPED
```

`PASS` is green. `FAIL(code)` and `TIMEOUT` are red. `DEPRECATED`, `ABORTED`, and `SKIPPED` are yellow so a glance distinguishes "something broke" from "skipped by design or by user". The same table (without colors) is written to `summary.log`.


## Logs

All logs are written to a timestamped directory:

```
logs/
  integration_test_2026-03-16_143644/
    summary.log                        # full results table
    calibration/
      bad_blood.log                    # stdout+stderr for this run
      bouncy_organ.log
      ...
    validation/
      bad_blood.log
      bouncy_organ.log
      ...
```

The `logs/` directory is gitignored.

### Reading a failed model's log

```bash
# Find the most recent test run
LATEST=$(ls -t logs/ | head -1)

# Read a specific model's log
cat "logs/$LATEST/calibration/bouncy_organ.log"
```


## Exit Code

The script itself exits:

- **`0`** — all runs passed.
- **`1`** — at least one run failed or timed out.

This makes it safe to use in CI or chain with `&&`:

```bash
bash run_integration_tests.sh --level cm && echo "All CM models passed"
```


## Examples

```bash
# Everything, all defaults (all models, calibration + validation, 30min timeout)
bash run_integration_tests.sh

# Just one model, just calibration
bash run_integration_tests.sh --models "counting_stars" --partitions "calibration"

# All PGM models with a longer timeout
bash run_integration_tests.sh --level pgm --timeout 3600

# All models except two, validation only
bash run_integration_tests.sh --exclude "purple_alien novel_heuristics" --partitions "validation"

# Exclude nothing (override the default purple_alien exclusion)
bash run_integration_tests.sh --exclude ""

# Use a different conda environment
bash run_integration_tests.sh --env views_r2darts2

# All baseline models only
bash run_integration_tests.sh --library baseline

# All stepshifter models at country-month level
bash run_integration_tests.sh --library stepshifter --level cm

# Multiple flags combined: specific models, one partition, custom timeout
bash run_integration_tests.sh --models "bad_blood counting_stars" --partitions "calibration" --timeout 600
```


## Important Details

- **Single shared environment**: Unlike each model's own `run.sh` (which creates/activates a per-model conda env), this script uses one environment for all models. All models must be installable into that environment. If a model needs packages that conflict with the shared env, it will fail.
- **`--exclude` replaces, not appends**: Passing `--exclude "foo"` means *only* `foo` is excluded — `purple_alien` is no longer excluded unless you include it: `--exclude "purple_alien foo"`.
- **Models run sequentially**: There is no parallelism. A full run of all models across 2 partitions can take many hours depending on model complexity and data fetch times.
- **Data is fetched live**: Each model's queryset pulls data from the VIEWS API at runtime. Network issues or API downtime will cause failures unrelated to model code.
- **Forecasting partition uses live time**: If you pass `--partitions "forecasting"`, the train/test ranges are computed from `ViewsMonth.now()`, so results depend on when you run.
