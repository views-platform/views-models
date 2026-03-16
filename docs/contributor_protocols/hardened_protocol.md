# The Hardened Protocol: ML Forecasting Governance

This document defines mandatory engineering and scientific standards for the views-models repository. Adherence is required for all contributions to guarantee forecast integrity and reproducibility.

---

## 1. Core Principles

### A. The Authority of Declarations (ADR-003)
**"Never infer; only trust declarations."**
All meaningful model properties (algorithm, level, targets, partition boundaries) must be explicitly declared in config files.
- **Prohibited:** Inferring model level from queryset name, algorithm from import path, or partition boundaries from model type.
- **Requirement:** If a property affects model identity or evaluation, it must be a mandatory key in the config files.

### B. The Fail-Loud Mandate (ADR-008)
**"A crash is a successful defense of scientific integrity."**
Silent failures, implicit fallbacks, and "best-effort" corrections are forbidden.
- **Requirement:** Missing config keys must cause test failures, not silent defaults.
- **Prohibited:** Using `warnings.filterwarnings("ignore")` to hide config-level problems (ML library warnings are acceptable to suppress).

### C. Partition Integrity
**"All models evaluate on the same temporal windows."**
Partition boundaries are defined once in `common/partitions.py` and shared by all models.
- **Requirement:** Every model's `config_partitions.py` must be a one-line import from `common.partitions`.
- **Prohibited:** Model-specific partition overrides, custom forecasting offsets, or hardcoded boundary values in individual model configs.
- **Rationale:** Divergent partitions make model evaluation metrics incomparable.

### D. Ensemble Ordering Discipline
**"CM before PGM."**
Country-month ensembles must complete before priogrid-month ensembles to enable reconciliation.
- **Requirement:** PGM ensembles declare `reconcile_with` referencing their CM counterpart.
- **Current enforcement:** Documented in `docs/monthly_run_guide.md` (not yet programmatic).

---

## 2. Contributor Requirements

### Adding a New Model
1. **Use the scaffold:** Run `python build_model_scaffold.py` to create the directory structure.
2. **Complete all configs:** Fill in all 6 config files with actual values — do not leave template defaults.
3. **Verify conventions:** Run `pytest tests/ -v` to confirm the new model passes all convention tests.
4. **Delegate partitions:** Ensure `config_partitions.py` contains only `from common.partitions import generate`.

### Adding a New Required Config Key
1. **Add to test expectations:** Update `REQUIRED_META_KEYS` or `REQUIRED_HP_KEYS` in `tests/test_config_completeness.py`.
2. **Add to all models:** Add the key with the correct value to all ~66 models.
3. **Verify:** `pytest tests/ -v` — all models must pass.
4. **Document:** Write an ADR if the key represents a significant architectural decision.

---

## 3. Mandatory Testing Taxonomy (ADR-005)

### Green Team (Correctness)
- Config files contain all required keys with valid values
- `time_steps` matches `len(steps)` where applicable
- Catalog generation uses `importlib`, not `exec()`

### Beige Team (Convention Drift)
- All models follow naming convention `^[a-z]+_[a-z]+$`
- All models have required files (`main.py`, `run.sh`, 6 config files)
- All models use the new CLI pattern (`ForecastingModelArgs`)
- All models delegate partitions to `common.partitions`

### Red Team (Adversarial) — Future Work
- Invalid config values don't propagate silently
- Broken queryset files don't crash catalog generation for other models
- Ensemble dependency chains are validated

---

## 4. Operational Invariants

- **Partition Centralization:** All models share `common/partitions.py`. No exceptions.
- **CLI Uniformity:** All models use `ForecastingModelArgs.parse_args()`. No explicit `wandb.login()`.
- **Config Completeness:** All required keys are present. Tests enforce this.
- **Catalog Safety:** Config loading uses `importlib.util`, not `exec()`.
