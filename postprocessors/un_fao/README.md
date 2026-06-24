# un_fao postprocessor

Delivers VIEWS data to the **UN FAO**. It fetches historical UCDP fatality
**actuals** from the VIEWS **data factory** (Zarr store), enriches them with
country/admin attribution, and uploads the result to the FAO Appwrite bucket
(served downstream by `views-faoapi`).

Created: 2025-10-17.

## What it produces

Historical actuals for three UCDP fatality targets at PRIO-GRID month level:

| descriptor feature | renamed to | UCDP source |
|---|---|---|
| `ged_sb_best` | `lr_ged_sb` | state-based |
| `ged_ns_best` | `lr_ged_ns` | non-state |
| `ged_os_best` | `lr_ged_os` | one-sided |

Data source is the **data factory** (not viewser) — see `configs/config_queryset.py`.
Current coverage region: **`africa_me_legacy`** (~13,110 cells). Going global to
`land_gaul` (64,736 cells) is tracked in issue #127.

## Prerequisites (datafactory)

The postprocessor reads the data factory's remote Zarr store, so the runtime
environment **must** have:

1. **`views-datafactory` installed** (provides the `datafactory_query` module):
   ```
   pip install 'views-datafactory @ git+https://github.com/views-platform/views-datafactory.git@development'
   ```
   If it is missing, `configs/config_queryset.py` fails loud at config load with
   these install instructions (it does not fall through to an opaque
   `ModuleNotFoundError`).
2. **`~/.netrc` credentials for the Zarr host `204.168.219.108`** on the machine
   that runs the postprocessor (see the `bright_starship` model README for the
   `.netrc` format). Without it, the Zarr fetch fails at runtime.

These are the same prerequisites the datafactory models (bright_starship,
shining_codex) need, but the postprocessor runs in its own context — verify them
in **its** runtime, not only a model env (issue #95).

## The ensemble reference

`configs/config_meta.py` carries an `ensemble` field. It does **not** select an
aggregation; it locates the `.env` credentials and tags uploaded files for the
**forecast** delivery path. It must reference a real ensemble (issue #77).

## Running

```
conda run -n views_pipeline python -m postprocessors.un_fao.main
```

This runs the full pipeline including the **Appwrite upload** to the production
forecasts bucket — do not run it casually. For a local, no-upload data check,
exercise `configs/config_queryset.py::generate()` + the data-factory fetch
directly (see `tests/test_un_fao_datafactory_equivalence.py`).
