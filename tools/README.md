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

## `credentials/`

What credentials this repo declares, and where the non-secret coordinates come from.
Never holds or emits a secret value — the one secret (`APPWRITE_DATASTORE_API_KEY`) stays
an operator slot per PLATFORM-001.

```bash
python tools/credentials/check_credentials.py                  # which keys am I missing?
python tools/credentials/registry_to_env.py <registry.toml>    # coordinates from the owned registry
```

`registry_to_env.py` is invoked by `postprocessors/un_fao/run.sh`; changing its path
changes production.

## `audit/`

Read-only verification passes. Each answers one yes/no about the repository and exits with
that answer, so they compose with the merge ritual and with CI.

```bash
bash tools/audit/shell_health.sh          # run.sh hygiene: shebangs, permissions, hardcoded paths
bash tools/audit/verify_committed.sh      # is the suite green on COMMITTED state, not your dirty tree?
python tools/audit/queryset_transforms.py # ADR-012: no queryset-level log transforms on targets
```

## `liveness/`

Is the platform alive on every input and output surface? See
[`liveness/README.md`](liveness/README.md) — it has its own contract
(`docs/CICs/LivenessChecks.md`).

```bash
python -m tools.liveness
```

---

### A note on this layout

The heading of this file — *"each subdirectory handles one responsibility"* — is a rule
that decays quietly. It was established when C-60 replaced a flat pile of scripts with
`catalogs/`, `partitions/` and `scaffold/`. By 2026-07-31 six loose files had accumulated
at `tools/` root again: the rule was written down, and nothing checked it. If you are
adding a tool here, it belongs in a directory named for its responsibility — and if no
existing one fits, that is a signal worth taking seriously, not a reason to drop the file
at the root.
