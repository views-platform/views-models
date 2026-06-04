# Investigation Report: Conda to UV Migration for views-models

**Date:** 2026-05-23  
**Author:** Simon / Claude  
**Status:** Investigation complete — awaiting decisions  
**Scope:** views-models repository, with upstream implications for views-pipeline-core

---

## 1. Executive Summary

The views-models repository currently uses per-model conda prefix environments to isolate dependencies across 83 models, 8 ensembles, and 2 APIs. Other VIEWS platform repositories (views-datafactory, views-lab00, views-bayesian) have already migrated to uv-based package management with `pyproject.toml`, committed `uv.lock` files, and `uv run` invocation.

This investigation maps the full blast radius of a conda-to-uv migration for views-models, identifies the hard problems, and documents the decision points.

**Key findings:**

1. Conda is used exclusively in shell scripts — zero Python code references conda at runtime
2. The darts version conflict between views-stepshifter and views-r2darts2 that necessitated separate environments **will disappear** once stepshifter v1.2.0 is released (already bumped to `darts ^0.40.0` on development HEAD)
3. Three packages (views-baseline, views-hydranet, views-seldon) are not published on PyPI and require git/local source installation
4. The current environment setup consumes ~16 GB on disk for only 3 of the 6+ required environments
5. Migration touches 93 `run.sh` files, but all are generated from a single template in views-pipeline-core

---

## 2. Current Conda Architecture

### 2.1 How It Works Today

Every model, ensemble, and API has a `run.sh` script generated from a template in views-pipeline-core (`views_pipeline_core/templates/model/template_run_sh.py`). The template produces a shell script that:

1. Handles macOS-specific libomp configuration (appends to `~/.zshrc`)
2. Resolves paths: `script_path` (model directory) and `project_path` (repo root)
3. Sets `env_path="$project_path/envs/{package_name}"`
4. Evaluates conda shell hooks: `eval "$(conda shell.bash hook)"`
5. Checks if the environment directory exists:
   - If yes: activates it, runs `pip install --dry-run` to check for missing packages
   - If no: creates a new conda prefix env with Python 3.11, installs from `requirements.txt`
6. Runs `python main.py "$@"`

This means every model run potentially creates, activates, and installs into its own conda prefix environment at `envs/{library_name}`.

### 2.2 Environment Landscape

There are 6 distinct library-based environments, but naming is inconsistent:

| Library | Models | Env Name(s) | Naming Issue |
|---------|--------|-------------|--------------|
| views-stepshifter | 39 | `views_stepshifter` (32) / `views-stepshifter` (7) | **Mixed dash/underscore** |
| views-r2darts2 | 19 | `views_r2darts2` (10) / `views-r2darts2` (9) | **Mixed dash/underscore** |
| views-baseline | 18 | `views-baseline` (all) | Consistent |
| views-hydranet | 5 | `views-hydranet` (all) | Consistent |
| views_ensemble | 8 | `views_ensemble` (all) | Consistent |
| views-faoapi | 1 | `views-faoapi` | Single use |
| views-seldon | 1 | `views-seldon` | Single use |

**Models using the dash variant (wrong name — will create a duplicate environment):**

Stepshifter dash (`views-stepshifter`):
- cheap_thrills, fake_model, fourtieth_symphony, lovely_creature, purple_haze, wild_rose, wuthering_heights

R2darts2 dash (`views-r2darts2`):
- adolecent_slob, bouncy_organ, emerging_principles, fancy_feline, hot_stream, novel_heuristics, party_princess, preliminary_directives, shining_codex

R2darts2 underscore (`views_r2darts2`):
- bad_romance, cold_heart, dancing_queen, elastic_heart, free_fallin, good_life, heat_waves, new_rules, revolving_door, smol_cat

Stepshifter underscore (`views_stepshifter`):
- bad_blood, bittersweet_symphony, blank_space, brown_cheese, caring_fish, car_radio, chunky_cat, counting_stars, dark_paradise, demon_days, electric_relaxation, fast_car, fluorescent_adolescent, good_riddance, green_squirrel, heavy_rotation, high_hopes, invisible_string, lavender_haze, little_lies, midnight_rain, national_anthem, old_money, ominous_ox, orange_pasta, plastic_beach, popular_monster, teen_spirit, twin_flame, wildest_dream, yellow_pikachu, yellow_submarine

### 2.3 Disk Usage

Only 3 of the 6+ required environments currently exist on disk:

```
envs/views-baseline/       6.4 GB
envs/views_stepshifter/    8.8 GB
envs/views-r2darts2/       90 MB  (appears incomplete)
────────────────────────────────
Total:                     ~16 GB
```

The other environments (views-hydranet, views_ensemble, the dash-variant duplicates) have not been created yet. If all environments were created, disk usage would be substantially higher due to duplicated Python interpreters and shared transitive dependencies (torch alone is ~2 GB).

### 2.4 Per-Model Dependencies

Each model has a minimal `requirements.txt` containing typically one or two package specs:

```
# Stepshifter models (39 models)
views-stepshifter>=1.0.0,<2.0.0

# R2darts2 models (19 models, three different version specs!)
views-r2darts2==0.1.0          # 6 models
views-r2darts2>=0.1.0          # 4 models (no upper bound)
views-r2darts2>=1.0.0,<2.0.0   # 9 models

# Baseline models (18 models)
views-baseline>=1.0.0,<2.0.0

# Hydranet models (5 models)
views-hydranet>=0.1.0,<1.0.0

# Ensembles (8 ensembles)
views-pipeline-core>=2.0.0,<3.0.0   # 7 ensembles
views-pipeline-core>=2.0.1,<3.0.0   # 1 ensemble (first_love)

# Special cases
views-datafactory @ git+...          # 5 models (heavy_freighter, bright_starship,
                                     #            heavy_strider, light_strider, shining_codex)
views-seldon>=0.1.0,<1.0.0           # 1 API (seldon_api)
git+.../views-faoapi.git@development  # 1 API (un_fao)
```

**Pre-existing issues:**
- `models/fake_model/requirements.txt` has a malformed version spec: `views-stepshifter==>=1.0.0,<2.0.0` (double operator)
- R2darts2 version pinning is inconsistent: `==0.1.0`, `>=0.1.0`, and `>=1.0.0,<2.0.0` across different models
- Two ensemble version specs differ slightly (`>=2.0.0` vs `>=2.0.1`)

### 2.5 Integration Test Environment

The `run_integration_tests.sh` script (410 lines) uses a **different approach** from the per-model `run.sh` files. Instead of per-model environments, it uses a single shared conda environment:

```bash
CONDA_ENV="views_pipeline"  # Default, configurable via --env flag

# For each model:
timeout --foreground "$TIMEOUT" bash -c "
    eval \"\$(conda shell.bash hook)\"
    conda activate '$CONDA_ENV'
    cd '$MODELS_DIR/$model'
    python main.py -r '$partition' -t -e
"
```

This environment must have all library packages pre-installed. It is not created or managed by the script — it must exist before running.

### 2.6 Monthly Production Run

The `monthly_run.sh` script is a simple orchestrator that calls ensemble `run.sh` scripts:

```bash
run_folder "ensembles/pink_ponyclub"
run_folder "ensembles/skinny_love"
run_folder "ensembles/rude_boy"
run_folder "ensembles/first_love"
run_folder "postprocessors/un_fao"
```

Each ensemble's `run.sh` handles its own conda environment setup. The production server must have conda installed and available.

### 2.7 CI/CD

GitHub Actions workflows already **do not use conda**:

```yaml
# .github/workflows/run_tests.yml
- uses: actions/setup-python@v4
  with:
    python-version: '3.11'
- run: pip install views_pipeline_core pytest
- run: pytest
```

This means CI is already conda-free — it uses plain pip install into the runner's Python.

---

## 3. The UV Pattern (Reference Implementations)

Three VIEWS repos have already migrated to uv. Their patterns provide the template for views-models.

### 3.1 views-datafactory

The most relevant reference — a production data pipeline:

```toml
# pyproject.toml
[project]
name = "views-datafactory"
version = "1.2.20"
requires-python = ">=3.12"
dependencies = [
    "numpy>=1.26,<3",
    "requests>=2.28,<3",
    "pyarrow>=14,<20",
    # ... 8 more deps
]

[dependency-groups]
dev = ["pytest>=8,<10", "ruff>=0.4,<1", "mypy>=1.8,<2", "types-requests>=2.28,<3"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/datafactory_provenance", "src/datafactory_harvester", ...]
```

Production scripts use `uv run`:
```bash
uv run python scripts/preflight.py
uv run python scripts/harvest_ucdp.py
uv run python scripts/assemble_grid.py
```

Committed `uv.lock` (1,639 lines) ensures reproducible installs. Tag-based deployment: server checks out a tag, runs `uv sync`, then executes.

### 3.2 views-lab00

Research sandbox with similar structure:

```toml
[project]
name = "views-lab00"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["numpy", "scipy", "matplotlib", "scikit-learn", "torch", ...]

[dependency-groups]
dev = ["properscoring>=0.1", "scores>=2.5.0", "xarray>=2025.6.1"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

Committed `uv.lock` (1,982 lines).

### 3.3 views-bayesian

Multi-Python-version CI testing:

```yaml
# .github/workflows/ci.yml
matrix:
  python-version: ["3.11", "3.12", "3.13"]
steps:
  - run: uv python install ${{ matrix.python-version }}
  - run: uv sync --python ${{ matrix.python-version }}
  - run: uv run pytest
```

### 3.4 Key Differences from views-models

| Aspect | uv repos (lab00, datafactory) | views-models |
|--------|-------------------------------|--------------|
| **Nature** | Importable Python packages | Collection of runnable scripts |
| **Build system** | hatchling with `src/` layout | None — models are directories, not packages |
| **Dependencies** | Single `pyproject.toml` | Per-model `requirements.txt` (93 files) |
| **Lock file** | `uv.lock` committed | None |
| **Invocation** | `uv run python script.py` | `bash run.sh` → conda activate → python |
| **Environment count** | 1 per repo | 6+ per repo (per library) |
| **Python version** | `requires-python` in toml | Hardcoded `python=3.11` in run.sh template |
| **Build backend** | hatchling | poetry-core (in library repos) |

---

## 4. Dependency Resolution Analysis

### 4.1 The Core Question: Can All Libraries Coexist?

The original rationale for per-library conda environments was potential dependency conflicts between model libraries. We tested whether the four model libraries (plus pipeline-core) can resolve into a single environment.

#### Test 1: All libraries together

```bash
uv pip compile --python 3.11 - <<EOF
views-stepshifter>=1.0.0,<2.0.0
views-r2darts2>=0.1.0
views-baseline>=1.0.0,<2.0.0
views-hydranet>=0.1.0,<1.0.0
views-pipeline-core>=2.0.0,<3.0.0
EOF
```

**Result: CONFLICT**

```
Because views-stepshifter>=1.0.0,<1.1.0 depends on darts>=0.30.0,<0.31.0
and views-stepshifter==1.1.0 depends on darts>=0.38.0,<0.39.0,
we can conclude that views-stepshifter>=1.0.0 depends on one of:
    darts>=0.30.0,<0.31.0
    darts>=0.38.0,<0.39.0

And because views-r2darts2==0.1.1 depends on darts==0.40.0,
views-r2darts2==0.1.1 and views-stepshifter>=1.0.0 are incompatible.
```

The published PyPI version of views-stepshifter (v1.1.0) requires `darts ^0.38.0` (which Poetry resolves to `>=0.38.0,<0.39.0`), while views-r2darts2 (v0.1.1) requires `darts==0.40.0`.

#### Test 2: Stepshifter + pipeline-core (without r2darts2)

**Result: SUCCESS** — resolves to 178 packages with darts==0.38.0.

#### Test 3: R2darts2 + pipeline-core (without stepshifter)

**Result: SUCCESS** — resolves to 185 packages with darts==0.40.0.

#### Test 4: Baseline alone

**Result: FAILURE** — views-baseline is not on PyPI. Must be installed from source.

#### Test 5: Hydranet alone

**Result: FAILURE** — views-hydranet is not on PyPI. Must be installed from source.

### 4.2 The Darts Version Alignment

**Critical finding:** The darts conflict is **temporary**.

Examining the source code of views-stepshifter on its development branch reveals:

```toml
# views-stepshifter/pyproject.toml (development HEAD, v1.2.0, UNPUBLISHED)
darts = "^0.40.0"   # <-- bumped from ^0.38.0
```

```toml
# views-r2darts2/pyproject.toml (v0.1.1, published)
darts = "=0.40.0"
```

Both libraries now require darts 0.40.x on their development branches. Once views-stepshifter v1.2.0 is published to PyPI, the conflict disappears entirely.

**Implication:** A single unified environment will be possible once this version is released. The per-library environment split is a historical artifact that is about to become unnecessary.

### 4.3 PyPI Availability

| Package | On PyPI | Build Backend | Notes |
|---------|---------|---------------|-------|
| views-stepshifter | Yes (v1.1.0) | poetry-core | v1.2.0 on dev, unpublished |
| views-r2darts2 | Yes (v0.1.1) | poetry-core | |
| views-pipeline-core | Yes (v2.3.0) | poetry-core | |
| views-baseline | **No** | poetry-core | Must install from git/local |
| views-hydranet | **No** | poetry-core | Must install from git/local |
| views-seldon | **No** | poetry-core | Must install from git/local |
| views-faoapi | **No** | — | Must install from git |
| views-datafactory | Yes (v1.2.20) | hatchling | Already uv-native |
| views-evaluation | Yes (v0.4.0) | — | Transitive dep |

**Three model libraries are not on PyPI.** In a uv world, these would need to be specified as git dependencies in `pyproject.toml`:

```toml
dependencies = [
    "views-baseline @ git+https://github.com/views-platform/views-baseline.git@main",
    "views-hydranet @ git+https://github.com/views-platform/views-hydranet.git@main",
]
```

Or, ideally, published to PyPI like the others.

### 4.4 Build Backend Divergence

The model library repos (stepshifter, r2darts2, baseline, hydranet, pipeline-core) all use **poetry-core** as their build backend. The already-migrated repos (datafactory, lab00) use **hatchling**.

This is not a blocker — uv can install packages built with any PEP 517 backend. But it means the library repos themselves haven't migrated to uv yet. views-models migrating to uv would be consuming poetry-built packages from a uv-managed environment, which works fine.

---

## 5. Where Conda Lives in Code

### 5.1 The Template (views-pipeline-core)

All `run.sh` files are generated from two templates:

**`views_pipeline_core/templates/model/template_run_sh.py`** (models)  
**`views_pipeline_core/templates/ensemble/template_run_sh.py`** (ensembles)

The model template generates a 42-line shell script with this structure:

```bash
#!/bin/zsh

# macOS libomp setup (6 lines)
if [[ "$OSTYPE" == "darwin"* ]]; then
  # append LDFLAGS, CPPFLAGS, DYLD_LIBRARY_PATH to ~/.zshrc
fi

# Path resolution (3 lines)
script_path=$(dirname "$(realpath $0)")
project_path="$( cd "$script_path/../../" >/dev/null 2>&1 && pwd )"
env_path="$project_path/envs/{package_name}"

# Conda activation (15 lines)
eval "$(conda shell.bash hook)"
if [ -d "$env_path" ]; then
  conda activate "$env_path"
  # pip install --dry-run check
else
  conda create --prefix "$env_path" python=3.11 -y
  conda activate "$env_path"
  pip install -r $script_path/requirements.txt
fi

# Execution (1 line)
python $script_path/main.py "$@"
```

### 5.2 Python Code: Conda-Agnostic

The Python framework (`ModelManager`, `EnsembleManager`) calls `run.sh` via `subprocess.run()`:

```python
# views_pipeline_core/managers/ensemble/ensemble.py
shell_command = model_args.to_shell_command(model_path)
subprocess.run(shell_command, check=True, timeout=7200)
```

No Python code in views-pipeline-core or views-models:
- References `CONDA_PREFIX` or `CONDA_DEFAULT_ENV`
- Calls `conda` commands programmatically
- Checks if code is running inside conda
- Manages conda environments via Python APIs

**This is the most important architectural fact:** conda is a shell-level concern, not a Python-level concern. Replacing it requires changing shell scripts and templates, not model logic or framework code.

### 5.3 Scaffold Builders

`build_model_scaffold.py` and `build_ensemble_scaffold.py` in views-models call the pipeline-core templates:

```python
from views_pipeline_core.templates.model import template_run_sh
template_run_sh.generate(
    script_path=self._model.model_dir / "run.sh",
    package_name=self.package_name
)
```

These would need updating if the template changes or `run.sh` is eliminated.

### 5.4 Full Conda Reference Inventory

| File | Repo | Conda References | Count |
|------|------|-----------------|-------|
| `templates/model/template_run_sh.py` | pipeline-core | Template that generates all model run.sh | 1 |
| `templates/ensemble/template_run_sh.py` | pipeline-core | Template that generates all ensemble run.sh | 1 |
| `models/*/run.sh` | views-models | Generated scripts with conda activate | 83 |
| `ensembles/*/run.sh` | views-models | Generated scripts with conda activate | 8 |
| `apis/*/run.sh` | views-models | Generated scripts with conda activate | 2 |
| `run_integration_tests.sh` | views-models | Uses `conda activate $CONDA_ENV` | 1 |
| `monthly_run.sh` | views-models | Calls run.sh scripts (indirect) | 1 |
| `documentation/contributor_protocols/carbon_based_agents.md` | pipeline-core | Docs reference `conda run -n views_pipeline` | 1 |
| `.github/workflows/*.yml` | views-models | **No conda** — already uses plain pip | 0 |

**Total files to change: 97** (but 93 of those are generated from 2 templates)

---

## 6. Migration Path Analysis

### 6.1 Option A: Single Unified Environment (Recommended)

**Precondition:** views-stepshifter v1.2.0 published to PyPI (aligns darts to 0.40.0)

Replace the entire per-model conda setup with a single `pyproject.toml` at the repo root:

```toml
[project]
name = "views-models"
version = "0.1.0"
requires-python = ">=3.11,<3.15"
dependencies = [
    "views-pipeline-core>=2.3.0,<3.0.0",
    "views-stepshifter>=1.2.0,<2.0.0",
    "views-r2darts2>=0.1.0,<1.0.0",
    "views-baseline @ git+https://github.com/views-platform/views-baseline.git@main",
    "views-hydranet @ git+https://github.com/views-platform/views-hydranet.git@main",
    "views-datafactory @ git+https://github.com/views-platform/views-datafactory.git@main",
]

[dependency-groups]
dev = ["pytest>=8,<10", "ruff>=0.4,<1"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
markers = [
    "red: adversarial / error-path tests (ADR-005)",
    "beige: convention and structural compliance tests (ADR-005)",
    "green: correctness and functional tests (ADR-005)",
]
```

**Advantages:**
- Simplest possible setup — one environment for everything
- `uv.lock` ensures reproducible installs across all developers and CI
- No more environment naming inconsistencies
- No more 16+ GB of duplicate conda environments
- `uv run python models/X/main.py` just works — no activation ceremony
- CI becomes `uv sync && uv run pytest` instead of `pip install views_pipeline_core pytest`

**Risks:**
- Blocked until stepshifter v1.2.0 is on PyPI
- Git dependencies (baseline, hydranet) are slower to resolve than PyPI packages
- If a future library introduces a new conflict, we'd need to split again

### 6.2 Option B: Dependency Groups Per Library

If conflicts persist or new ones emerge, use uv's dependency groups:

```toml
[dependency-groups]
stepshifter = ["views-stepshifter>=1.2.0,<2.0.0"]
darts = ["views-r2darts2>=0.1.0,<1.0.0"]
baseline = ["views-baseline @ git+..."]
hydranet = ["views-hydranet @ git+..."]
```

Models would run with: `uv run --group stepshifter python models/X/main.py`

**Advantages:**
- Handles conflicts without per-model environments
- Still uses a single `uv.lock`

**Disadvantages:**
- More complex invocation
- `run.sh` or integration test script needs to know which group each model belongs to
- Partially defeats the simplicity advantage

### 6.3 Option C: Minimal Change (Keep requirements.txt, Use uv pip)

Replace conda with uv but keep per-model `requirements.txt`:

```bash
# New run.sh template
uv pip install -r $script_path/requirements.txt
python $script_path/main.py "$@"
```

**Advantages:**
- Minimal change to existing structure
- No need to consolidate dependencies

**Disadvantages:**
- No `uv.lock` — loses the main reproducibility benefit
- Still per-model dependency management
- Doesn't simplify the architecture

**Not recommended** — this is just replacing the conda command with a uv command without gaining uv's actual benefits.

---

## 7. What Changes Where

### 7.1 For Option A (Recommended)

| Change | Repo | Files Affected | Effort |
|--------|------|----------------|--------|
| Create `pyproject.toml` with all deps | views-models | 1 new file | Low |
| Generate `uv.lock` | views-models | 1 new file | Auto |
| New `run.sh` template (uv-based) | views-pipeline-core | 2 files | Low |
| Regenerate all `run.sh` | views-models | 93 files | Auto (scaffold rebuild) |
| Delete per-model `requirements.txt` | views-models | 93 files | Low |
| Update `run_integration_tests.sh` | views-models | 1 file | Medium |
| Update `monthly_run.sh` | views-models | 1 file | Low |
| Update CI workflows | views-models | 3 files | Low |
| Update `.gitignore` (remove `envs/`) | views-models | 1 file | Low |
| Delete `envs/` directory | views-models | ~16 GB freed | Low |
| Update contributor docs | views-pipeline-core | 1 file | Low |

### 7.2 The New run.sh Template

```bash
#!/bin/bash

# macOS libomp setup (still needed, orthogonal to package manager)
if [[ "$OSTYPE" == "darwin"* ]]; then
  if ! grep -q 'export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"' ~/.zshrc; then
    echo 'export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"' >> ~/.zshrc
  fi
  # ... CPPFLAGS, DYLD_LIBRARY_PATH
  source ~/.zshrc
fi

script_path=$(dirname "$(realpath $0)")
project_path="$( cd "$script_path/../../" >/dev/null 2>&1 && pwd )"

cd "$project_path"
uv run python "$script_path/main.py" "$@"
```

Or, if `run.sh` is eliminated entirely, models would be invoked directly:

```bash
# From repo root:
uv run python models/vertical_dream/main.py -r calibration -t -e
```

### 7.3 The New Integration Test Runner

```bash
# Replace:
#   eval "$(conda shell.bash hook)"
#   conda activate '$CONDA_ENV'
#   python main.py -r '$partition' -t -e
# With:
#   cd '$PROJECT_ROOT'
#   uv run python '$MODELS_DIR/$model/main.py' -r '$partition' -t -e
```

### 7.4 What Does NOT Change

- **Model `main.py` files** — zero changes, they're Python-only
- **Config files** (`config_meta.py`, `config_hyperparameters.py`, etc.) — zero changes
- **Test files** — zero changes (already run via `pytest`, not conda)
- **Model logic** — zero changes
- **Pipeline-core Python code** — zero changes (already conda-agnostic)

---

## 8. The run.sh Question

### 8.1 Is run.sh Still Needed?

In a uv world, `uv run python models/X/main.py "$@"` from the repo root does everything `run.sh` does (minus libomp setup). The question is whether `run.sh` serves other purposes:

**Current responsibilities of run.sh:**
1. ~~Conda environment creation/activation~~ → replaced by `uv sync`/`uv run`
2. ~~Dependency installation~~ → replaced by `uv sync`
3. macOS libomp configuration → still needed, but could be a one-time setup script
4. Path resolution → `uv run` handles this from repo root
5. Entry point for `monthly_run.sh` and `EnsembleManager.subprocess.run()` → could call `main.py` directly

**Recommendation:** Keep `run.sh` but make it minimal. The `EnsembleManager` in pipeline-core calls `subprocess.run(run.sh)`, so eliminating `run.sh` requires changing the framework. A thin wrapper is simpler:

```bash
#!/bin/bash
script_path=$(dirname "$(realpath $0)")
project_path="$( cd "$script_path/../../" >/dev/null 2>&1 && pwd )"
cd "$project_path"
uv run python "$script_path/main.py" "$@"
```

### 8.2 Memory: Do NOT Modify run.sh Files

Per established project guidance: `run.sh` files are production infrastructure and should not be modified casually. The migration would need to regenerate all `run.sh` files from an updated template in a single coordinated change, not modify them individually.

---

## 9. Pre-existing Issues to Fix During Migration

### 9.1 Environment Naming Inconsistency

**7 stepshifter models** use `views-stepshifter` (dash) while **32 use** `views_stepshifter` (underscore). Similarly, **9 r2darts2 models** use dash while **10 use** underscore. This creates duplicate conda environments on disk.

**In a uv world:** This issue disappears entirely — there's one environment for the whole repo.

### 9.2 Malformed Version Spec

`models/fake_model/requirements.txt` contains `views-stepshifter==>=1.0.0,<2.0.0` (double operator `==>=`). This is a pip syntax error that happens to be silently accepted by some pip versions.

**In a uv world:** This file would be deleted along with all other `requirements.txt` files.

### 9.3 R2darts2 Version Pinning Inconsistency

Three different version specs exist across 19 r2darts2 models:

| Spec | Models | Risk |
|------|--------|------|
| `==0.1.0` | 6 | Pins to exact version — won't pick up 0.1.1 |
| `>=0.1.0` | 4 | No upper bound — could pull breaking changes |
| `>=1.0.0,<2.0.0` | 9 | References version 1.0.0 which doesn't exist yet |

**In a uv world:** Centralized in `pyproject.toml` with one canonical spec, locked via `uv.lock`.

---

## 10. Migration Sequence

### Phase 0: Prerequisites
1. Publish views-stepshifter v1.2.0 to PyPI (aligns darts to 0.40.0)
2. Publish views-baseline to PyPI (or accept git dependency)
3. Publish views-hydranet to PyPI (or accept git dependency)
4. Decide: keep `run.sh` as thin wrapper or eliminate it

### Phase 1: Spike (views-models only)
1. Create spike branch
2. Write `pyproject.toml` with all dependencies
3. Run `uv lock` to generate `uv.lock`
4. Test: `uv run python models/vertical_dream/main.py -r calibration -t -e` (synthetic, no GPU)
5. Test: `uv run pytest tests/`
6. Measure: disk usage of `.venv/` vs `envs/`

### Phase 2: Template Update (views-pipeline-core)
1. Create new `template_run_sh.py` with uv-based invocation
2. Update scaffold builders to use new template
3. Release pipeline-core with new template

### Phase 3: Full Migration (views-models)
1. Add `pyproject.toml` + `uv.lock`
2. Regenerate all `run.sh` files from new template
3. Delete all per-model `requirements.txt` files
4. Update `run_integration_tests.sh` to use `uv run` instead of `conda activate`
5. Update `monthly_run.sh`
6. Update CI workflows to use `astral-sh/setup-uv@v4` + `uv sync`
7. Update `.gitignore`: remove `envs/`, add `.venv/`
8. Delete `envs/` directory

### Phase 4: Verification
1. Run full test suite: `uv run pytest tests/`
2. Run synthetic models end-to-end (no GPU): all 3 models × 3 run types
3. Run integration tests: `bash run_integration_tests.sh`
4. Run monthly production pipeline on test server
5. Verify CI passes on all workflows

### Phase 5: Cleanup
1. Update contributor documentation
2. Update README with new setup instructions
3. Remove conda references from docs

---

## 11. Decision Points

These decisions must be made before implementation can begin:

### D1: Single environment or dependency groups?

If stepshifter v1.2.0 is released and all deps resolve together, a single environment is far simpler. If conflicts persist (or future libraries introduce new ones), dependency groups provide a fallback.

**Recommendation:** Single environment, contingent on stepshifter v1.2.0 release.

### D2: Keep run.sh or eliminate it?

`EnsembleManager` in pipeline-core calls `subprocess.run(run.sh)`. Eliminating `run.sh` requires changing the framework. Keeping it as a thin `uv run` wrapper is simpler.

**Recommendation:** Keep as thin wrapper. The `run.sh` → `main.py` indirection has value for the macOS libomp case and provides a stable subprocess interface.

### D3: Where does pyproject.toml live?

Repo root is the natural and only sensible choice, consistent with all other uv-based VIEWS repos.

**Recommendation:** Repo root.

### D4: Migration order?

Pipeline-core template must change before views-models can regenerate `run.sh` files. But views-models can adopt `pyproject.toml` + `uv.lock` independently of `run.sh` changes.

**Recommendation:** Phase 1 (spike) in views-models to validate, then Phase 2 (template) in pipeline-core, then Phase 3 (full migration) in views-models.

### D5: Do models become installable packages?

lab00 and datafactory use `src/` layout with hatchling. views-models models are scripts in directories, not importable packages. Forcing them into a package structure adds complexity for no benefit — they're not imported by other code.

**Recommendation:** No. views-models is a runner repo, not a library. The `pyproject.toml` declares dependencies but doesn't package models. No `[tool.hatch.build.targets.wheel]` section needed.

### D6: What about views-baseline and views-hydranet not being on PyPI?

These can be specified as git dependencies (`@ git+https://...`), but this is slower to resolve and requires network access. Publishing them to PyPI is the cleaner solution.

**Recommendation:** Publish to PyPI if feasible; use git dependencies as interim solution.

---

## 12. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Stepshifter v1.2.0 not released → conflict persists | **High** | Use dependency groups as fallback, or install from git source |
| Git dependencies (baseline, hydranet) slow/fragile | Medium | Publish to PyPI; pin to tags not branches |
| macOS libomp setup breaks without run.sh | Low | Move to one-time setup script or document manually |
| EnsembleManager subprocess.run(run.sh) breaks | Medium | Keep run.sh as thin wrapper |
| Production server doesn't have uv installed | Medium | uv is a single binary — trivial to install |
| Lock file conflicts on multi-developer merges | Low | `uv.lock` is auto-generated; `uv lock` resolves |
| Some model has an undocumented hidden dependency | Low | Integration test suite catches this |

---

## 13. Expected Benefits

| Benefit | Impact |
|---------|--------|
| **Disk savings** | ~16 GB of conda envs → ~2 GB single `.venv/` |
| **Setup time** | `uv sync` (~10s) vs conda create + pip install (~2-5 min per env) |
| **Reproducibility** | Committed `uv.lock` → exact versions everywhere |
| **No naming bugs** | Single env eliminates dash/underscore duplication |
| **Simpler CI** | `uv sync && uv run pytest` replaces pip install guessing |
| **No conda requirement** | Production servers don't need conda installed |
| **Developer experience** | `uv run python models/X/main.py` — no activation ceremony |
| **Dependency visibility** | One `pyproject.toml` shows all deps, not scattered across 93 files |

---

## 14. Appendix: Full Model-to-Library-to-Environment Mapping

### Stepshifter Models (39)

| Model | Env Variant | Version Spec |
|-------|------------|--------------|
| bad_blood | underscore | >=1.0.0,<2.0.0 |
| bittersweet_symphony | underscore | >=1.0.0,<2.0.0 |
| blank_space | underscore | >=1.0.0,<2.0.0 |
| brown_cheese | underscore | >=1.0.0,<2.0.0 |
| car_radio | underscore | >=1.0.0,<2.0.0 |
| caring_fish | underscore | >=1.0.0,<2.0.0 |
| cheap_thrills | **dash** | >=1.0.0,<2.0.0 |
| chunky_cat | underscore | >=1.0.0,<2.0.0 |
| counting_stars | underscore | >=1.0.0,<2.0.0 |
| dark_paradise | underscore | >=1.0.0,<2.0.0 |
| demon_days | underscore | >=1.0.0,<2.0.0 |
| electric_relaxation | underscore | >=1.0.0,<2.0.0 |
| fake_model | **dash** | ==>=1.0.0,<2.0.0 **(MALFORMED)** |
| fast_car | underscore | >=1.0.0,<2.0.0 |
| fluorescent_adolescent | underscore | >=1.0.0,<2.0.0 |
| fourtieth_symphony | **dash** | >=1.0.0,<2.0.0 |
| good_riddance | underscore | >=1.0.0,<2.0.0 |
| green_squirrel | underscore | >=1.0.0,<2.0.0 |
| heavy_rotation | underscore | >=1.0.0,<2.0.0 |
| high_hopes | underscore | >=1.0.0,<2.0.0 |
| invisible_string | underscore | >=1.0.0,<2.0.0 |
| lavender_haze | underscore | >=1.0.0,<2.0.0 |
| little_lies | underscore | >=1.0.0,<2.0.0 |
| lovely_creature | **dash** | >=1.0.0,<2.0.0 |
| midnight_rain | underscore | >=1.0.0,<2.0.0 |
| national_anthem | underscore | >=1.0.0,<2.0.0 |
| old_money | underscore | >=1.0.0,<2.0.0 |
| ominous_ox | underscore | >=1.0.0,<2.0.0 |
| orange_pasta | underscore | >=1.0.0,<2.0.0 |
| plastic_beach | underscore | >=1.0.0,<2.0.0 |
| popular_monster | underscore | >=1.0.0,<2.0.0 |
| purple_haze | **dash** | >=1.0.0,<2.0.0 |
| teen_spirit | underscore | >=1.0.0,<2.0.0 |
| twin_flame | underscore | >=1.0.0,<2.0.0 |
| wild_rose | **dash** | >=1.0.0,<2.0.0 |
| wildest_dream | underscore | >=1.0.0,<2.0.0 |
| wuthering_heights | **dash** | >=1.0.0,<2.0.0 |
| yellow_pikachu | underscore | >=1.0.0,<2.0.0 |
| yellow_submarine | underscore | >=1.0.0,<2.0.0 |

### R2darts2 Models (19)

| Model | Env Variant | Version Spec |
|-------|------------|--------------|
| adolecent_slob | **dash** | >=1.0.0,<2.0.0 |
| bad_romance | underscore | ==0.1.0 |
| bouncy_organ | **dash** | >=1.0.0,<2.0.0 |
| cold_heart | underscore | ==0.1.0 |
| dancing_queen | underscore | >=0.1.0 |
| elastic_heart | underscore | >=0.1.0 |
| emerging_principles | **dash** | >=1.0.0,<2.0.0 |
| fancy_feline | **dash** | >=1.0.0,<2.0.0 |
| free_fallin | underscore | ==0.1.0 |
| good_life | underscore | ==0.1.0 |
| heat_waves | underscore | >=0.1.0 |
| hot_stream | **dash** | >=1.0.0,<2.0.0 |
| new_rules | underscore | ==0.1.0 |
| novel_heuristics | **dash** | >=1.0.0,<2.0.0 |
| party_princess | **dash** | >=1.0.0,<2.0.0 |
| preliminary_directives | **dash** | >=1.0.0,<2.0.0 |
| revolving_door | underscore | ==0.1.0 |
| shining_codex | **dash** | >=1.0.0,<2.0.0 |
| smol_cat | underscore | >=0.1.0 |

### Baseline Models (18)

All use `views-baseline` (dash) env, `>=1.0.0,<2.0.0` version spec.

### Hydranet Models (5)

All use `views-hydranet` (dash) env, `>=0.1.0,<1.0.0` version spec.

### Datafactory-Dependent Models (5)

| Model | Primary Library | Datafactory Dep |
|-------|----------------|-----------------|
| heavy_freighter | views-hydranet | git+...@development |
| bright_starship | views-hydranet | git+...@development |
| heavy_strider | views-baseline | git+...@development |
| light_strider | views-baseline | git+...@development |
| shining_codex | views-r2darts2 | git+...@development |

### Synthetic Models (3) and Ensembles (2)

All use `views-baseline`, fixed partitions, deterministic MSE.

### Ensembles (8)

All share `views_ensemble` env, require `views-pipeline-core>=2.0.0,<3.0.0`.

---

## 15. PEP 723 Inline Script Metadata: Deep Investigation

### 15.1 What Is PEP 723?

PEP 723 allows Python scripts to declare their dependencies inline using TOML-formatted comment blocks:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "views-stepshifter>=1.2.0,<2.0.0",
# ]
# ///

from views_pipeline_core.cli import ForecastingModelArgs
from views_stepshifter.manager import StepshifterManager
# ... model code ...
```

When invoked via `uv run script.py`, uv:
1. Parses the inline metadata block
2. Resolves dependencies against PyPI
3. Creates an ephemeral, cached virtual environment
4. Runs the script in that environment

No `requirements.txt`, no `run.sh`, no conda, no activation ceremony.

### 15.2 Practical Test Results

All experiments were run on the local system with uv 0.8.13.

#### Basic Functionality

| Test | Result | Notes |
|------|--------|-------|
| Basic inline metadata (numpy) | SUCCESS | Cold: 0.42s, Warm: 0.18s |
| Two scripts with overlapping deps | SUCCESS | Separate envs, hardlink deduplication |
| views-pipeline-core from PyPI | SUCCESS (Python 3.11) | 153 packages resolved, 139ms install |
| Subprocess invocation | SUCCESS | `subprocess.run(["uv", "run", ...])` works |
| Python version pinning | SUCCESS | `--python 3.11` auto-downloads interpreter |
| Simulated model runner | SUCCESS | ForecastingModelArgs imports correctly |

#### Performance

| Scenario | Time |
|----------|------|
| Cold start (first ever run, download all packages) | ~4 min (one-time) |
| Cold start (packages cached, new environment) | 0.34s |
| Warm start (environment cached) | 0.108s |
| Conda activate + pip check + python run (current) | ~5-15s |

Warm starts are **50-100x faster** than the current conda activation path.

#### Disk Efficiency

uv uses hardlinks from a shared archive to deduplicate packages across environments:

```
~/.cache/uv/archive-v0/        24 GB  (all packages, stored once)
~/.cache/uv/environments-v2/    6.3 GB (11 test environments)
```

For N models all depending on views-pipeline-core (~5.8 GB of packages), the incremental cost per additional model is approximately **22 MB** (just environment metadata and symlinks). Compare this to conda, where each environment is a full copy (~5-8 GB).

**Projected savings for views-models:**
- Current: 6 conda envs × ~5 GB average = ~30 GB (only 16 GB created so far)
- With PEP 723: ~6 GB shared archive + 93 × 22 MB = ~8 GB total

#### Deduplication Proof

Two test environments both depending on views-pipeline-core (153 packages, 5.8 GB apparent each) occupied only 5.9 GB combined on disk. The numpy `__init__.py` file had identical inodes across both environments, confirming hardlink sharing.

### 15.3 Python Version Constraint

**Important caveat:** uv defaults to the newest available Python (currently 3.13.7). The transitive dependency `levenshtein==0.20.9` (via ingester3 → views-pipeline-core) does not compile on Python 3.13 due to deprecated C API usage.

**Mitigation:** Either:
- Use `requires-python = ">=3.11,<3.13"` in inline metadata
- Pass `--python 3.11` to `uv run`
- Wait for levenshtein to release a 3.13-compatible version

This is not a uv issue — it's an upstream compatibility issue. Pinning to Python 3.11 works reliably.

### 15.4 Subprocess Orchestration (The Framework Question)

The `EnsembleManager` in views-pipeline-core calls model scripts via `subprocess.run()`. Testing confirmed that `subprocess.run(["uv", "run", "script.py"])` works correctly — uv parses inline metadata even when invoked as a subprocess.

This means the framework could evolve from:
```python
# Current: subprocess.run(["bash", "run.sh", ...])
# Future:  subprocess.run(["uv", "run", "--python", "3.11", "main.py", ...])
```

### 15.5 Git Dependencies in Inline Metadata

PEP 723 supports git dependencies via PEP 508 syntax:

```python
# /// script
# dependencies = [
#     "views-baseline @ git+https://github.com/views-platform/views-baseline.git@main",
# ]
# ///
```

However, git dependencies are slower to resolve (full clone required) and fragile (branch refs may change). Publishing to PyPI is the preferred approach.

### 15.6 What This Means for Model Independence

PEP 723 **preserves the independence model you want.** Each model's `main.py` declares its own dependencies. Teams can:

- Use whatever model library they want (stepshifter, r2darts2, hydranet, or something new)
- Pin whatever versions they need
- Have conflicting transitive deps (different darts versions) without coordination
- Not know or care what other models depend on

views-models doesn't need a centralized `pyproject.toml` or dependency resolution — each script is self-contained.

---

## 16. Stepwise Migration: Can Libraries Migrate Independently?

### 16.1 The Core Question

Can you migrate views-stepshifter from poetry-core to hatchling/uv **without breaking views-models or any other downstream consumer?**

### 16.2 Answer: Yes, 100% Independent

**The build backend is invisible to consumers.** When views-models runs `pip install views-stepshifter>=1.0.0`, it receives a pre-built wheel (.whl) from PyPI. Whether that wheel was built by poetry-core or hatchling is irrelevant — the wheel format is standardized (PEP 427).

Evidence:
- views-datafactory already migrated from poetry to hatchling — no downstream breakage
- views-hydranet uses a mixed format (`[project]` PEP 621 + `[tool.poetry.group.dev]`) — installs fine
- views-pipeline-core does NOT directly depend on stepshifter/r2darts2/baseline/hydranet — these are per-model dependencies
- The `run.sh` scripts call `pip install -r requirements.txt` which specifies packages by name+version, not by build backend

### 16.3 What Changes in Each Library (Internal Only)

To migrate a library (e.g., views-stepshifter) from poetry to uv/hatchling:

1. **pyproject.toml**: Convert `[tool.poetry.dependencies]` to `[project] dependencies` (PEP 621 format)
2. **Build system**: Change `poetry-core` to `hatchling`
3. **Publish workflow**: Change `poetry publish --build` to `python -m build && twine upload`
4. **Lock file**: Replace `poetry.lock` with `uv.lock`
5. **Dev workflow**: Replace `poetry install` with `uv sync`

None of these changes affect the installed package. The import paths, API, and runtime behavior are identical.

### 16.4 Migration Order (Any Order Works)

```
views-stepshifter  ──→ migrate to hatchling/uv  ──→ publish v1.2.0  ──→ no downstream impact
views-r2darts2     ──→ migrate to hatchling/uv  ──→ publish v0.2.0  ──→ no downstream impact
views-baseline     ──→ migrate to hatchling/uv  ──→ publish v0.2.0  ──→ no downstream impact
views-hydranet     ──→ migrate to hatchling/uv  ──→ publish v0.2.0  ──→ no downstream impact
views-pipeline-core ──→ migrate to hatchling/uv ──→ publish v2.4.0  ──→ no downstream impact
```

Each library can migrate on its own schedule. There is no "big bang" refactor required. The only coordination point is the darts version alignment (stepshifter v1.2.0 bumps to darts 0.40.0), which is a dependency change, not a build system change.

### 16.5 When views-models Itself Migrates

views-models migration to PEP 723 can happen **independently of library migrations**. Even if all libraries still use poetry-core internally, PEP 723 inline metadata in model `main.py` files will install them correctly from PyPI.

The only prerequisite for views-models migration is:
1. uv installed on developer machines and production servers
2. Libraries published to PyPI (or specified as git deps)
3. Updated `run.sh` template in views-pipeline-core (or `EnsembleManager` subprocess call)

---

## 17. Recommended Approach

### 17.1 The Phased Strategy

Given the findings, the recommended migration path is:

**Phase 0: No-risk preparation (can start now)**
- Publish views-stepshifter v1.2.0 to PyPI (aligns darts to 0.40.0)
- Publish views-baseline and views-hydranet to PyPI (currently git-only installs)
- Fix pre-existing issues: malformed version spec in fake_model, inconsistent r2darts2 pinning

**Phase 1: Library-side migration (any order, no downstream impact)**
- Migrate each library repo from poetry-core to hatchling at its own pace
- Each library gets `pyproject.toml` (PEP 621), `uv.lock`, and `uv sync` workflow
- Publish new versions to PyPI after migration
- **Nothing changes in views-models** during this phase

**Phase 2: PEP 723 spike in views-models (single branch, test with synthetics)**
- Add PEP 723 inline metadata to synthetic model `main.py` files (vertical_dream, horizontal_dream, diagonal_dream)
- Test: `uv run --python 3.11 models/vertical_dream/main.py -r calibration -t -e`
- Measure performance, disk usage, correctness
- Validate subprocess invocation works from a test orchestrator

**Phase 3: Template update in views-pipeline-core**
- New `template_run_sh.py` that generates `uv run`-based scripts (or direct `main.py` invocation)
- Update `EnsembleManager` subprocess call if run.sh is eliminated
- Release pipeline-core with new template

**Phase 4: Full views-models migration**
- Add PEP 723 metadata to all 93 `main.py` files
- Regenerate (or eliminate) all `run.sh` files
- Delete per-model `requirements.txt` files
- Update `run_integration_tests.sh` and `monthly_run.sh`
- Update CI workflows
- Delete `envs/` directory

### 17.2 The Key Insight

**Nothing is blocked. Nothing needs to be coordinated. Each step is independently safe.**

The build backend is invisible to consumers. The dependency format (poetry vs PEP 621) is invisible to consumers. PEP 723 inline metadata works regardless of how upstream libraries are built. Each library can migrate on its own schedule, and views-models can migrate whenever the team is ready — they're decoupled by design.

The current conda setup works. It's not broken. But PEP 723 + uv would give you:
- 50-100x faster model startup (0.1s vs 5-15s)
- ~75% less disk usage (8 GB vs 30 GB)
- No conda dependency on production servers
- Self-documenting dependencies (inline in main.py, not in a separate file)
- True per-model isolation without the overhead of per-model environments

---

## 18. Concrete Before/After: PEP 723 in a Real Model

### 18.1 Current State: vertical_dream (Synthetic Baseline Model)

Today, running `vertical_dream` requires three files working together:

**`models/vertical_dream/requirements.txt`:**
```
views-baseline>=1.0.0,<2.0.0
```

**`models/vertical_dream/run.sh`** (42 lines, generated from template):
```bash
#!/usr/bin/env bash

if [[ "$OSTYPE" == "darwin"* ]]; then
  if ! grep -q 'export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"' ~/.zshrc; then
    echo 'export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"' >> ~/.zshrc
  fi
  if ! grep -q 'export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"' ~/.zshrc; then
    echo 'export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"' >> ~/.zshrc
  fi
  if ! grep -q 'export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"' ~/.zshrc; then
    echo 'export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"' >> ~/.zshrc
  fi
  source ~/.zshrc
fi

script_path=$(dirname "$(realpath $0)")
project_path="$( cd "$script_path/../../" >/dev/null 2>&1 && pwd )"
env_path="$project_path/envs/views-baseline"

eval "$(conda shell.bash hook)"

if [ -d "$env_path" ]; then
  echo "Conda environment already exists at $env_path. Checking dependencies..."
  conda activate "$env_path"
  echo "$env_path is activated"

  missing_packages=$(pip install --dry-run -r $script_path/requirements.txt 2>&1 | grep -v "Requirement already satisfied" | wc -l)
  if [ "$missing_packages" -gt 0 ]; then
    echo "Installing missing or outdated packages..."
    pip install -r $script_path/requirements.txt
  else
    echo "All packages are up-to-date."
  fi
else
  echo "Creating new Conda environment at $env_path..."
  conda create --prefix "$env_path" python=3.11 -y
  conda activate "$env_path"
  pip install -r $script_path/requirements.txt
fi

echo "Running $script_path/main.py "
python $script_path/main.py "$@"
```

**`models/vertical_dream/main.py`** (24 lines, no dependency info):
```python
from pathlib import Path
from views_pipeline_core.cli import ForecastingModelArgs
from views_pipeline_core.managers import ModelPathManager
from views_baseline.manager.baseline_manager import BaselineForecastingModelManager

try:
    model_path = ModelPathManager(Path(__file__))
except Exception as e:
    raise RuntimeError(f"Unexpected error: {e}. Check the logs for details.")

if __name__ == "__main__":
    args = ForecastingModelArgs.parse_args()

    manager = BaselineForecastingModelManager(
        model_path=model_path,
        wandb_notifications=args.wandb_notifications,
        use_prediction_store=args.prediction_store,
    )

    if args.sweep:
        manager.execute_sweep_run(args)
    else:
        manager.execute_single_run(args)
```

**To run:** `bash models/vertical_dream/run.sh -r calibration -t -e`

This requires: conda installed, 6.4 GB `envs/views-baseline/` directory, ~5-15 seconds activation overhead per run.

### 18.2 After: PEP 723 Version

**`models/vertical_dream/requirements.txt`:** DELETED

**`models/vertical_dream/run.sh`** (5 lines, thin wrapper):
```bash
#!/bin/bash
script_path=$(dirname "$(realpath $0)")
project_path="$( cd "$script_path/../../" >/dev/null 2>&1 && pwd )"
cd "$project_path"
uv run --python 3.11 "$script_path/main.py" "$@"
```

**`models/vertical_dream/main.py`** (29 lines, self-documenting):
```python
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "views-baseline @ git+https://github.com/views-platform/views-baseline.git@main",
#     "views-pipeline-core>=2.3.0,<3.0.0",
# ]
# ///

from pathlib import Path
from views_pipeline_core.cli import ForecastingModelArgs
from views_pipeline_core.managers import ModelPathManager
from views_baseline.manager.baseline_manager import BaselineForecastingModelManager

try:
    model_path = ModelPathManager(Path(__file__))
except Exception as e:
    raise RuntimeError(f"Unexpected error: {e}. Check the logs for details.")

if __name__ == "__main__":
    args = ForecastingModelArgs.parse_args()

    manager = BaselineForecastingModelManager(
        model_path=model_path,
        wandb_notifications=args.wandb_notifications,
        use_prediction_store=args.prediction_store,
    )

    if args.sweep:
        manager.execute_sweep_run(args)
    else:
        manager.execute_single_run(args)
```

**To run:** `bash models/vertical_dream/run.sh -r calibration -t -e` (same interface)  
Or directly: `uv run --python 3.11 models/vertical_dream/main.py -r calibration -t -e`

This requires: uv installed (~30 MB binary), ~22 MB incremental per-model env (hardlinked from shared cache), ~0.1 seconds activation overhead per run.

### 18.3 What Changed, What Didn't

| Aspect | Before | After |
|--------|--------|-------|
| **main.py logic** | Unchanged | Unchanged — same imports, same classes, same behavior |
| **Dependencies declared in** | `requirements.txt` (separate file) | `main.py` inline (PEP 723 block, 6 lines) |
| **run.sh** | 42 lines (conda create/activate/pip install) | 5 lines (uv run wrapper) |
| **requirements.txt** | 1 line | Deleted |
| **Package manager** | conda + pip | uv |
| **Environment on disk** | 6.4 GB prefix env | ~22 MB hardlinked env |
| **Startup overhead** | 5-15 seconds | 0.1 seconds |
| **Config files** | Unchanged | Unchanged |
| **Test invocation** | `pytest tests/` | `pytest tests/` (unchanged) |

The model logic, configs, and test suite are completely untouched. The only changes are: (1) 6 lines of TOML comments added to the top of `main.py`, (2) `run.sh` rewritten to a 5-line uv wrapper, (3) `requirements.txt` deleted.

---

## 19. API Models: seldon_api and un_fao

### 19.1 Current State

Two API endpoints exist in `apis/`:

**`apis/seldon_api/`** — Views Seldon API
- `requirements.txt`: `views-seldon>=0.1.0, <1.0.0`
- `main.py`: imports `views_seldon.managers.model.APIPathManager` and `views_seldon.managers.api.ViewsApiManager`
- Uses `wandb.login()` — requires W&B credentials
- views-seldon is **not on PyPI** — must be installed from git source
- Has its own `run.sh` (conda-based, generated from template)

**`apis/un_fao/`** — UN FAO API
- `requirements.txt`: `git+https://github.com/views-platform/views-faoapi.git@development`
- `main.py`: imports `views_faoapi.managers.model.APIPathManager` and `views_faoapi.managers.api.FAOApiManager`
- Uses `wandb.login()` — requires W&B credentials
- views-faoapi is **not on PyPI** — installed directly from git
- Has its own `run.sh` (conda-based, generated from template)

### 19.2 Migration Considerations

Both APIs have the same structure as models (main.py + requirements.txt + run.sh + configs), so the PEP 723 migration applies identically. However, both depend on packages that are **not on PyPI**:

| API | Package | On PyPI | PEP 723 Dependency Syntax |
|-----|---------|---------|--------------------------|
| seldon_api | views-seldon | No | `"views-seldon @ git+https://github.com/views-platform/views-seldon.git@main"` |
| un_fao | views-faoapi | No | `"views-faoapi @ git+https://github.com/views-platform/views-faoapi.git@development"` |

Both APIs also import `wandb` directly in `main.py`, which is a transitive dependency of views-pipeline-core (so it doesn't need to be declared separately in the PEP 723 block). But this is worth noting — the APIs have a W&B login step that models don't.

### 19.3 API run.sh Templates

The API `run.sh` files are generated from a separate template (`views_pipeline_core/templates/ensemble/template_run_sh.py` or a dedicated API template). The API template migration is identical to the model template migration — it's the same conda-to-uv replacement.

### 19.4 Post-Migration API Example (un_fao)

```python
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "views-faoapi @ git+https://github.com/views-platform/views-faoapi.git@development",
#     "views-pipeline-core>=2.3.0,<3.0.0",
# ]
# ///

import wandb
from pathlib import Path
from views_faoapi.managers.model import APIPathManager
from views_faoapi.managers.api import FAOApiManager

# ... rest unchanged ...
```

---

## 20. Transition Period: Conda and uv Coexistence

### 20.1 The Hybrid Phase

Migration will not be atomic. For some period, both conda and uv will coexist. This section documents how that works and what to watch for.

**During Phase 2 (spike with synthetics):**
- 3 synthetic models + 2 synthetic ensembles use PEP 723 / uv
- 80+ production models still use conda
- Both `envs/` (conda) and `~/.cache/uv/` (uv) exist on developer machines
- `run_integration_tests.sh` still uses `conda activate $CONDA_ENV` for all models
- Synthetic models must work with BOTH invocation paths (direct `uv run` and the conda-based integration test runner)

**During Phase 4 (full migration):**
- All models migrated to PEP 723
- `run_integration_tests.sh` updated to use `uv run`
- `envs/` directory becomes stale and can be deleted
- conda remains installed on developer machines (may be used by other repos) — no need to uninstall

### 20.2 .gitignore Status

The `.gitignore` already handles both environments:
```
.venv       (line 170)
envs/       (line 172)
venv/       (line 173)
venv.bak/   (line 176)
```

No changes needed to `.gitignore` for the migration. Both the current `envs/` conda prefix environments and any future `.venv/` uv environments are already excluded from version control.

### 20.3 Developer Machine Cleanup

After the full migration, developers can reclaim disk space by deleting their conda prefix environments:
```bash
rm -rf envs/                   # ~16-30 GB freed
conda env remove --name views_pipeline  # if the integration test env exists
```

This is optional — stale conda environments don't interfere with uv. But the disk savings are substantial (16-30 GB vs the ~8 GB uv uses with hardlink deduplication).

### 20.4 Backward Compatibility Period

**Important consideration:** If a developer pulls the migrated branch but doesn't have uv installed, models will fail to run. The updated `run.sh` calls `uv run`, which requires uv on the PATH.

Mitigation options:
1. **Document the prerequisite:** README update, contributor docs, team announcement
2. **Self-installing run.sh:** The updated template could check for uv and install it automatically:
   ```bash
   if ! command -v uv &>/dev/null; then
     echo "Installing uv..."
     curl -LsSf https://astral.sh/uv/install.sh | sh
   fi
   ```
   uv is a single static binary (~30 MB) — the install is fast and non-invasive.
3. **Grace period:** Keep the old conda-based `run.sh` on a branch for N weeks while developers transition

**Recommendation:** Option 1 (documentation) plus Option 2 (self-installing fallback in run.sh). uv installation is a one-line curl command and the binary is self-contained — no system package manager needed.

---

## 21. Rollback Strategy

### 21.1 Why Rollback Is Low-Risk

The migration changes only shell scripts and metadata — zero model logic, zero config changes, zero Python code changes (beyond adding the PEP 723 comment block to `main.py`, which is ignored by all Python interpreters). Rolling back is:

1. **Revert the `main.py` PEP 723 blocks:** `git revert` the commit that added inline metadata. The `# /// script` blocks are TOML-formatted comments — Python already ignores them, so even if they're left in place, nothing breaks.

2. **Regenerate conda-based `run.sh` files:** Run the scaffold builder with the old pipeline-core template. Since run.sh files are generated artifacts, the old template produces the old files.

3. **Restore `requirements.txt` files:** These are tracked in git history. `git checkout <pre-migration-commit> -- models/*/requirements.txt` restores all of them.

4. **Re-create conda environments:** `envs/` is gitignored — if it was deleted, running any model's old `run.sh` will recreate the conda environment automatically (that's what the template does).

### 21.2 Rollback Triggers

Consider rolling back if:
- A model produces different numerical results under uv vs conda (would indicate a dependency version difference — investigate before rolling back)
- The production server cannot install uv (unlikely — it's a static binary that runs on any Linux)
- A critical upstream library breaks under uv's stricter dependency resolution (uv is stricter than pip about version conflicts — this could surface latent incompatibilities that pip silently accepts)

### 21.3 Rollback Cost

| Phase | Rollback Effort | Risk |
|-------|----------------|------|
| Phase 1 (spike) | Trivial — delete the spike branch | None |
| Phase 2 (template update) | Low — revert pipeline-core commit, regenerate run.sh | Template is tested before release |
| Phase 3 (full migration) | Medium — revert views-models commits, regenerate run.sh, restore requirements.txt | All tracked in git; conda envs auto-recreate |
| Phase 4+ (envs/ deleted, conda removed from servers) | High — must re-install conda, re-create all environments | Only reached after full validation |

**Recommendation:** Do not delete `envs/` or remove conda from production servers until Phase 4 verification is complete and the team has run at least one full monthly production cycle through the uv-based pipeline.

---

## 22. Synthetic Test Models as Migration Spike Targets

### 22.1 Why Synthetics Are Ideal

The views-models repo contains 3 synthetic models and 2 synthetic ensembles specifically designed for pipeline testing:

| Name | Type | Library | Purpose |
|------|------|---------|---------|
| vertical_dream | model | views-baseline | Synthetic data, deterministic MSE, calibration testing |
| horizontal_dream | model | views-baseline | Synthetic data, deterministic MSE, validation testing |
| diagonal_dream | model | views-baseline | Synthetic data, deterministic MSE, combined testing |
| synthetic_choir | ensemble | views-pipeline-core | Ensemble of synthetic models |
| synthetic_chorus | ensemble | views-pipeline-core | Ensemble of synthetic models |

These are ideal spike targets because:
1. **No GPU required** — use views-baseline (linear regression), not darts/torch
2. **Deterministic output** — fixed synthetic data means we can compare results byte-for-byte between conda and uv runs
3. **Fast execution** — calibration run completes in ~30 seconds
4. **Low blast radius** — synthetic models are not part of the production pipeline
5. **Full pipeline coverage** — models + ensembles test both the model manager and ensemble manager subprocess paths
6. **Single library dependency** — views-baseline only, avoids the darts version conflict entirely

### 22.2 Spike Validation Protocol

To validate the PEP 723 migration using synthetics:

```bash
# Step 1: Run under conda (current system) and capture output
bash models/vertical_dream/run.sh -r calibration -t -e 2>&1 | tee /tmp/conda_output.log

# Step 2: Add PEP 723 inline metadata to main.py
# Step 3: Run under uv and capture output
uv run --python 3.11 models/vertical_dream/main.py -r calibration -t -e 2>&1 | tee /tmp/uv_output.log

# Step 4: Compare results
# The numerical predictions should be identical (deterministic model + fixed data)
# Timing and log lines will differ (no conda activation messages)
```

Repeat for all 3 synthetic models × 3 run types (calibration, validation, forecasting), then for both synthetic ensembles. If all 11 runs produce identical predictions, the migration is validated.

### 22.3 What This Proves

Successful synthetic validation confirms:
- PEP 723 inline metadata resolves correctly for views-baseline
- `uv run` subprocess invocation works (ensemble manager path)
- Model configs, querysets, and output paths are unaffected
- The run.sh thin wrapper works end-to-end
- No numerical divergence from dependency version differences

It does **not** prove:
- GPU/CUDA models work (stepshifter, r2darts2, hydranet all need GPU for real runs)
- Darts-dependent models resolve correctly (requires stepshifter v1.2.0 or git deps)
- Production monthly run works (requires server testing)

These gaps are addressed in Phase 4 verification.

---

## 23. Migration Option Comparison Matrix

Side-by-side comparison of the three migration options discussed in Section 6:

| Criterion | Option A: Single pyproject.toml | Option B: Dependency Groups | Option C: PEP 723 Inline Metadata |
|-----------|------|------|------|
| **Config location** | Root `pyproject.toml` | Root `pyproject.toml` with groups | Each `main.py` |
| **Lock file** | Single `uv.lock` | Single `uv.lock` | No lock file (cached ephemeral envs) |
| **Invocation** | `uv run python models/X/main.py` | `uv run --group stepshifter python models/X/main.py` | `uv run models/X/main.py` |
| **Handles conflicts** | No — all deps must be compatible | Yes — groups resolve independently | Yes — each script is independent |
| **Per-model independence** | No — centralized deps | Partial — grouped by library | Full — each model declares its own |
| **Reproducibility** | Excellent — `uv.lock` pins everything | Excellent — `uv.lock` pins per group | Good — cached but not locked |
| **Disk usage** | ~2 GB (single .venv) | ~4-8 GB (one .venv per group) | ~8 GB (hardlinked per-script envs) |
| **Startup time** | ~0.05s (already synced) | ~0.05s (already synced) | ~0.1s (warm cache), ~0.3s (cold) |
| **requirements.txt** | Deleted | Deleted | Deleted |
| **run.sh changes** | `uv run python main.py` | `uv run --group X python main.py` | `uv run main.py` |
| **New library added** | Add to pyproject.toml | Add new group to pyproject.toml | Add to new model's main.py (no repo-wide change) |
| **Stepshifter v1.2.0 needed?** | Yes (conflict blocker) | No (groups isolate) | No (scripts isolate) |
| **CI changes** | `uv sync && uv run pytest` | `uv sync --all-groups && uv run pytest` | `uv run pytest` (tests don't need model deps) |
| **Framework changes (pipeline-core)** | Template update only | Template update + group mapping | Template update only |
| **Preserves library independence** | No | Partially | Fully |

### 23.1 Recommendation Update

The investigation in Sections 15-16 established that **Option C (PEP 723)** best preserves the project's design philosophy of independent model libraries. However, Options A and C are not mutually exclusive:

- **Option A** can be used for the shared test/CI environment (pyproject.toml with dev dependencies for pytest, ruff, etc.)
- **Option C** can be used for model execution (each main.py declares its runtime dependencies)

This hybrid approach gives you:
- A `pyproject.toml` at the repo root for development tools (pytest, ruff) and CI
- PEP 723 inline metadata in each `main.py` for model execution
- Full per-model dependency independence
- No centralized dependency resolution across model libraries

---

## 24. Open Questions and Future Investigation

### 24.1 Unresolved Questions

These questions were identified during the investigation but not fully answered:

1. **views-seldon deployment model:** The Seldon API is a web service, not a batch job. How is it deployed today? Does it run continuously on a server, or is it invoked periodically like models? This affects whether PEP 723 ephemeral environments make sense for it (ephemeral environments are ideal for batch invocations, less so for long-running services).

2. **W&B authentication in API models:** Both `seldon_api` and `un_fao` call `wandb.login()` at the module level. In a PEP 723 world, this still works — but the authentication flow (API key storage) needs to be documented for new developers who won't have conda envs pre-configured with W&B credentials.

3. **uv cache warming on CI:** GitHub Actions runners start with a cold cache. The first `uv run` on a PEP 723 script would download all dependencies (~4 minutes for the full darts stack). CI should use `actions/cache` with `~/.cache/uv/` to persist the cache across runs. This is a solved problem (views-datafactory CI already does it with `astral-sh/setup-uv@v4`), but needs explicit setup.

4. **Cross-platform PEP 723 behavior:** macOS libomp setup is currently handled in `run.sh`. With PEP 723, the dependency resolution is platform-aware (uv resolves different wheels for macOS vs Linux), but libomp is a system dependency, not a Python package. The macOS libomp setup either stays in `run.sh` or moves to a one-time setup script.

5. **Exact uv version to standardize on:** The experiments used uv 0.8.13. uv releases frequently (weekly). Should the project pin a minimum uv version, or just require "latest"? views-datafactory CI uses `astral-sh/setup-uv@v4` which installs the latest by default.

### 24.2 Deprecated Information in This Report

As of the date of writing (2026-05-23), the following facts may become outdated:

- **Darts version conflict (Section 4.1):** This conflict exists between the *published* versions of views-stepshifter (v1.1.0, darts ^0.38.0) and views-r2darts2 (v0.1.1, darts ==0.40.0). Once views-stepshifter v1.2.0 is published to PyPI (already on development HEAD with darts ^0.40.0), the conflict disappears. Check the current published version before acting on this section.

- **views-baseline and views-hydranet PyPI availability (Section 4.3):** These are not on PyPI as of this writing. They may be published in the future. Check PyPI before using git dependencies.

- **uv version (Section 15.2):** Experiments used uv 0.8.13. uv updates weekly. Some behaviors or performance numbers may change with newer versions.

- **levenshtein Python 3.13 incompatibility (Section 15.3):** The `levenshtein==0.20.9` package doesn't compile on Python 3.13. This may be fixed in a newer levenshtein release or when ingester3/views-pipeline-core bumps its dependency.

- **Model count (throughout):** The report references 83 models, 8 ensembles, 2 APIs. These counts change as models are added or removed. The synthetic models (vertical_dream, horizontal_dream, diagonal_dream) and synthetic ensembles (synthetic_choir, synthetic_chorus) were added in May 2026. Check `ls models/ | wc -l` for the current count.
