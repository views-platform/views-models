# ADR-011: Partition Boundary Semantics

**Status:** Accepted
**Date:** 2026-04-06
**Deciders:** Simon (project maintainer)
**Informed:** All contributors

---

## Context

All models, ensembles, extractors, and postprocessors share a common temporal partitioning scheme that divides the VIEWS data timeline into calibration, validation, and forecasting windows. These boundaries are critical for cross-model comparability — ensemble aggregation is only meaningful if constituent models were evaluated on the same data.

Partition boundaries are classified as **Tier 1 — Stable** (ADR-004). Each entity has its own self-contained `config_partitions.py` (ADR-002), resulting in the same values duplicated across 73+ files. Prior to this ADR:

- The canonical values existed only implicitly in test assertions
- No migration tooling existed for coordinated updates
- Two files had silently drifted to non-standard values (discovered 2026-04-06)

---

## Decision

### Canonical Partition Boundaries

The single source of truth for partition boundaries is `meta/partitions.json`:

```json
{
  "calibration": {"train": [121, 444], "test": [445, 492]},
  "validation": {"train": [121, 492], "test": [493, 540]},
  "forecasting_offset": -1,
  "forecasting_origin": 121,
  "steps_default": 36
}
```

### ViewsMonth ID Mapping

| ViewsMonth ID | Calendar Date | Role |
|---------------|---------------|------|
| 121 | January 1990 | Training origin (all partitions) |
| 444 | December 2016 | Calibration train end |
| 445 | January 2017 | Calibration test start |
| 492 | December 2020 | Calibration test end / Validation train end |
| 493 | January 2021 | Validation test start |
| 540 | December 2024 | Validation test end |

Formula: `ViewsMonth = (year - 1980) * 12 + month`

### Invariants

1. **No train/test overlap:** For each partition, `train_end < test_start`
2. **Cross-model identity:** All models and ensembles MUST use identical boundaries for calibration and validation to ensure comparability
3. **Forecasting is rolling:** `ViewsMonth.now().id - 1` ensures forecasting always uses the most recent complete month
4. **Origin is fixed:** Training always starts at ViewsMonth 121 (January 1990)

### Override Mechanism

A `config_partitions.py` file may use non-standard boundaries if it contains a declaration:

```python
# PARTITION_OVERRIDE: <reason for deviation>
```

**Consequences of declaring an override:**
- The partition consistency test (`test_config_partitions.py`) will **warn** but not fail
- The bump tool (`tools/partitions/bump.py`) will **skip** the file with a warning
- The override and its rationale are visible in test output and bump logs
- **After a bump, override files retain pre-bump values** — they must be reviewed and updated manually

Undeclared deviations (non-standard values without the marker) are treated as **test failures**.

### Annual Partition Bump

Partitions advance forward by 12 month_ids annually, typically in June/July after UCDP releases calibrated annual data. The bump tool handles this:

```bash
python -m tools.partitions.bump               # dry run (default)
python -m tools.partitions.bump --execute     # apply changes
```

The tool enforces:
- **7 structural invariants:** train start anchored at 121, test windows = 48 months, partition chaining
- **Temporal plausibility:** validation test end cannot exceed Dec (current_year - 1) — blocks double-bumps and future-dated partitions
- **Pre-flight check:** all files must match current canonical before bumping
- **Post-write verification:** every file re-read and compared after writing
- **Atomic writes:** tempfile + os.replace to prevent corruption
- **JSONL lockfile** with git state (commit, branch, dirty flag) in `meta/partition_bump_*.jsonl`

Files with `PARTITION_OVERRIDE` are skipped — the bump summary lists them for manual review.

### Manual Migration (sync without advancing)

To sync all files to canonical values without advancing partitions:

```bash
python -m tools.partitions.bump --bump 0 --execute
```

---

## Consequences

### Positive
- Single source of truth eliminates ambiguity about canonical values
- Bump tool reduces a 100-file manual edit to a single command with safety checks
- Override mechanism permits legitimate deviations while making them visible
- Data leakage test (`test_train_before_test`) catches off-by-one boundary errors
- Temporal plausibility check prevents partitions from extending beyond available UCDP data

### Negative
- `meta/partitions.json` is a new file contributors must know about
- The bump tool is regex-based and assumes the current file structure; a major structural change to `config_partitions.py` would require updating `tools/partitions/fileops.py`
- Override files must be manually reviewed after each bump — they are not automatically updated

---

## References

- [ADR-002](002_topology.md) — Self-contained config files (why duplication exists)
- [ADR-004](004_evolution.md) — Partition boundaries as Tier 1 — Stable
- `meta/partitions.json` — Canonical partition values
- `tools/partitions/bump.py` — Annual bump tool
- `tools/partitions/domain.py` — Partition invariants and temporal validation
- `tools/partitions/fileops.py` — Shared parser for config_partitions.py files
- `tests/test_config_partitions.py` — Enforcement tests
- `tests/test_bump_partitions.py` — Bump tool unit tests
