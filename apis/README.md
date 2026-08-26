# `apis/` — deployment launchers for VIEWS serving APIs

This directory holds **API deployment launchers**, a run-bearing category distinct
from `models/` and `postprocessors/`. Each `apis/<name>/` is a **thin launcher**:
it creates a conda env, `pip install`s an external `views-*` API package from
GitHub, and runs it via `run.sh` → `main.py`. **The real service code lives in the
external package, not here.** A launcher follows the same config convention as a
model (`configs/config_meta.py`, `configs/config_deployment.py`, `main.py`,
`run.sh`, `requirements.txt`), with `config_meta.algorithm = "API"`.

## What's here

| Launcher | Installs + runs | What that package is |
|---|---|---|
| `apis/un_fao/` | `views-faoapi` (`run.sh` → `pip install git+…/views-faoapi.git`) | FastAPI REST service that serves VIEWS conflict predictions to the UN FAO (historical + probabilistic forecasts at PGM / country / GAUL levels; reads from Appwrite) |
| `apis/seldon_api/` | `views-seldon` (`run.sh` → `pip install git+…/views-seldon.git`) | A Seldon-style serving API (wraps `seldonapi_postprocessor`) |

## The distinction that matters (so you don't get lost)

For `un_fao` specifically there are **three** different `un_fao`-named things across
the platform — they are *not* duplicates, they are different roles:

- **`apis/un_fao/`** (here) — the **launcher** for the FAO serving API.
- **`postprocessors/un_fao/`** (this repo) — the **producer**: builds + uploads the
  FAO delivery (historical actuals + forecast) to Appwrite. See its README.
- **`views-faoapi`** (separate repo) — the **package**: the actual FastAPI service the
  launcher installs, and the holder of the Appwrite **secrets** (`views-faoapi/.env`).

## Credentials

A launcher does **not** store secrets. The launched API reads its credentials from
the external package's own `.env` (e.g. `views-faoapi/.env`). views-models stays
secret-free. (See `postprocessors/un_fao/README.md` → "Credential topology" and
`reports/un_fao_delivery_{prerun,postrun}_postmortem.md` for the full Appwrite map.)

## Status

These launchers are early. `apis/un_fao/` on `development` is a minimal stub; a
fuller build-out (multi-worker uvicorn, endpoint docs) exists on an unmerged
feature branch (`sweep_week_dylan`) — revive-vs-defer is a maintainer decision, but
**do not delete** the stub (it's the intended FAO-serving entrypoint, not dead code).

## Running

```
./run.sh -r <run_type>        # bootstraps envs/<pkg>, installs the package, runs main.py
# or, if the env exists:
python main.py <args>
```
