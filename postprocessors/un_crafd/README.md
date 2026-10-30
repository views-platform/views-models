# un_crafd — the CRAF'd delivery launcher

Runs the CRAF'd producer: reads a `chrome_box` forecast from the shared
`production_forecasts` shelf, builds the ADR-013 wire, and (once armed) uploads it to
`crafd_bucket` for **views-crafdapi**.

**Not the sibling of `postprocessors/un_fao/` any more.** It was — both read
`rusty_bucket` — until 2026-10-28, when this delivery was repointed at `chrome_box`,
because `rusty_bucket` is FAO's and must not go to CRAF'd as a stopgap (Simon's ruling,
2026-10-22). Same three targets, same coverage; a different source and a different
destination.

`chrome_box` is a *reconciled* pgm forecast: it reconciles against `chrome_crate`'s cm
country forecast during its own ensemble run, so what lands on the shelf is already
reconciled. `chrome_crate` is an input to that, not a second thing this launcher reads —
and it could not be, since a postprocessor config carries exactly one source
(`deliveries/status.py:188`).

## The upload is disarmed

`deliveries/un_crafd.py` declares `intent = paused(...)`, so `wire_upload_enabled` derives
to `False` and the manager never constructs a partner store. A run stages artifacts under
`data/generated/wire_contract/<run_id>/` and makes **zero store calls**.

⚠️ **This section said exactly that while the declaration said `live`.** From
2026-08-14 to 2026-10-28 `intent` was `live(since=2026-08-14)`, and this paragraph went
on claiming the upload was disarmed. ⛔ And it DID fire: 111 uploads completed on
2026-08-14 20:36:25, logged under this directory's own `wandb/run-20260814_202751-g06spd7j/`.
The earlier claim that "nothing fired only because `monthly_run.sh:179` runs `un_fao` and
not this launcher" was false. When the two could disagree, read the declaration:
`upload_armed()` derives from `intent`, and prose derives from nobody.

Arming now waits on three things, each named with its reason in
`deliveries/un_crafd.py`'s docstring: views-pipeline-core#562, views-models#647 and
views-crafdapi#180. To arm it, change `intent` to `live(since=...)` in
`deliveries/un_crafd.py` — not here.

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
