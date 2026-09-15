# un_crafd — the CRAF'd delivery launcher

Runs the CRAF'd producer: reads a `rusty_bucket` forecast from the shared
`production_forecasts` shelf, builds the ADR-013 wire, and (once armed) uploads it to
`crafd_bucket` for **views-crafdapi**.

The sibling of `postprocessors/un_fao/`. Same source ensemble, same three targets, same
coverage — a different destination, and nothing else, today.

## The upload is disarmed

`deliveries/un_crafd.py` declares `intent = paused(...)`, so `wire_upload_enabled` derives
to `False` and the manager never constructs a partner store. A run stages artifacts under
`data/generated/wire_contract/<run_id>/` and makes **zero store calls**.

Arming is views-crafdapi's D5 (their #45), after the D4 dry run. To arm it, change
`intent` to `live(since=...)` in `deliveries/un_crafd.py` — not here.

## What is declared where

| fact | declared in | how this launcher gets it |
|---|---|---|
| source ensemble | `deliveries/un_crafd.py` `send` | `declared_source("un_crafd")` |
| coverage region | `deliveries/un_crafd.py` `coverage` | `declared_coverage("un_crafd")` |
| armed or not | `deliveries/un_crafd.py` `intent` | `upload_armed("un_crafd")` |
| Appwrite coordinates | the Appwrite Seam Contract registry | `tools/credentials/platform_env.sh` |

Nothing above is typed in `configs/`. That is ADR-019 and ADR-021, and it is why cloning
this directory for a third partner cannot inherit CRAF'd's region by accident.

## Running it

```bash
bash postprocessors/un_crafd/run.sh
```

The delivery protocol — registry check, conda, the #294 capability assertion, the
environment load — lives in `tools/launcher/postprocessor.sh` (ADR-022). This directory's
`run.sh` supplies only the conda environment name and the views-postprocessing pin.

Needs `views-datafactory` installed and a `~/.netrc` entry for the Zarr host
`204.168.219.108` for the historical leg.
