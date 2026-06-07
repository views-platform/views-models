# Tools

Operational tooling for the views-models repository. Each subdirectory handles one responsibility.

## `catalogs/`

Generates the model and ensemble catalog tables in README.md and per-model README files. Runs automatically in CI when model configs change.

```bash
python tools/catalogs/create_catalogs.py      # regenerate README catalog tables
python tools/catalogs/update_readme.py        # regenerate per-model READMEs
python tools/catalogs/generate_features_catalog.py  # feature catalog (manual)
```

CI workflow: `.github/workflows/update_catalogs.yml`

## `partitions/`

Annual partition boundary management. Advances calibration and validation time windows when UCDP releases new data.

```bash
python -m tools.partitions.bump               # dry run (default)
python -m tools.partitions.bump --execute     # apply the bump
```

See [ADR-011](../docs/ADRs/011_partition_semantics.md) for partition semantics.

## `scaffold/`

Creates new models, ensembles, and packages from templates. Run when adding a new model to the repository.

```bash
python tools/scaffold/build_model_scaffold.py       # new model
python tools/scaffold/build_ensemble_scaffold.py    # new ensemble
python tools/scaffold/build_package_scaffold.py     # new package
```

## `audit_shell_health.sh`

Checks all `run.sh` files for common issues (missing shebang, wrong permissions, hardcoded paths).

```bash
bash tools/audit_shell_health.sh
```
