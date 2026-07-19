# tools/liveness — are our forecasts live?

One command that answers, with raw facts, whether the VIEWS forecasting
system is alive on every input and output surface. Built as epic
[#238](https://github.com/views-platform/views-models/issues/238) after the
2026-07-19 episode in which nobody — human or AI — could check whether the
forecasts were live, and un-encoded conventions produced false alarms.

```bash
conda run -n views_pipeline python -m tools.liveness
```

That prints one raw-facts block per surface and exits with the worst code
across all of them. Each surface also runs alone:

```bash
conda run -n views_pipeline python -m tools.liveness.old_api
conda run -n views_pipeline python -m tools.liveness.datafactory_input
conda run -n views_pipeline python -m tools.liveness.appwrite_store
conda run -n views_pipeline python -m tools.liveness.unfao_delivery
conda run -n views_pipeline python -m tools.liveness.wandb_execution
conda run -n views_pipeline python -m tools.liveness.vpn_store
```

## Exit codes (uniform across every surface)

| Code | Meaning |
|------|---------|
| 0 | Healthy — **or a truthful SKIP**: missing credentials/package/VPN is a fact about *your environment*, not a failure of the surface |
| 1 | Reachable but stale / idle / not serving — needs attention |
| 2 | Unreachable — the surface cannot be observed at all |

The aggregate runner (`python -m tools.liveness`) exits with the **worst**
per-surface code and contains crashes: one broken check prints an
UNREACHABLE fact and code 2, and never hides the other surfaces.

## The surfaces and their verdicts

### `old_api` — the public API (`api.viewsforecasting.org`)
Is the newest published fatalities run fresh, and does it actually serve rows?

- `LIVE_FRESH` — newest run's data-cutoff month is within budget (≤ 2 months
  behind the current calendar month = 1 month publication lag + 1 grace).
- `LIVE_STALE` — listed but too old; `months_behind` says how far.
- `LIVE_NOT_SERVING` — run is listed but returned no rows for its first
  forecast month.
- `UNREACHABLE`.

### `datafactory_input` — the datafactory zarr input store
Does observed input coverage reach what this repo's canonical partitions
require? The requirement is **derived from `meta/partitions.json` at run
time** (max test-window end), so every partition bump re-arms the check —
this automates the register C-96 tripwire.

- `INPUT_FRESH` — live `last_valid_month_id` ≥ required (margin reported).
- `INPUT_STALE` — partitions outrun observed coverage: validation-tail
  "actuals" would be zero-fill, not observations. Do not trust validation
  metrics until this is green again.
- `SKIP_NO_PACKAGE` (no `datafactory_query` installed) / `UNREACHABLE`.

### `appwrite_store` — the internal Appwrite prediction shelf
Is anything landing on the `production_forecasts` bucket, and does the REAL
metadata collection exist?

- `STORE_ACTIVE` — newest file ≤ 45 days old (server-side
  `orderDesc($createdAt)` query — never the 25-per-page default listing,
  which produced a false-idle verdict on 2026-07-19).
- `STORE_IDLE` — nothing new in 45 days.
- `SKIP_NO_CREDENTIALS` / `UNREACHABLE`.

### `unfao_delivery` — the FAO partner bucket (`unfao_bucket`)
When did FAO last receive anything, per stream (`forecast_dataset_*` and
`historical_dataset_*` judged independently)?

- `DELIVERING` — both streams ≤ 45 days.
- `DELIVERY_STALLED` — at least one stream is `STALLED` or
  `NEVER_DELIVERED` (per-stream verdicts in the facts).
- `SKIP_NO_CREDENTIALS` / `UNREACHABLE`.

### `wandb_execution` — did the team compute this cycle?
Latest **finished** forecasting run per monthly ensemble
(`pink_ponyclub`, `skinny_love`, `rude_boy`, `first_love` — hand-encoded
mirror of `monthly_run.sh`; update both when the roster changes).

- `EXECUTION_CURRENT` — every ensemble `COMPUTED` within 40 days.
- `EXECUTION_STALE` — any ensemble `NOT_COMPUTED`/`NEVER_RUN`.
- `SKIP_NO_CREDENTIALS` (no `api.wandb.ai` in `~/.netrc`) / `UNREACHABLE`.

### `vpn_store` — the legacy Postgres store (`gjoll.muspelheim.local`)
Are computed runs uploaded to the PRIO-internal store (possibly awaiting
public promotion)? Host resolves **only on the PRIO VPN**.

- `STORE_FRESH` / `STORE_STALE` — same run-name parser and freshness budget
  as `old_api`.
- `VPN_REQUIRED` — host unresolvable: the truthful off-VPN verdict, never a
  false red.
- `SKIP_NO_PACKAGE` (no `views_forecasts` installed) / `UNREACHABLE`.

## Conventions encoded here (each exactly once, with receipts)

- **Run naming** (`tools/liveness/old_api.py`): official grammar
  `fatalities{gen}_{yyyy}_{mm}_t{seq}` where `{yyyy}_{mm}` is the
  **data-cutoff month** — "the last data that informs a given run"
  (views_api wiki) — NOT the execution month. Execution/publication happens
  ~1 month later. Misreading this caused the 2026-07-19 false
  "production stalled" alarm. `month_id = (year − 1980) × 12 + month`.
- **The real Appwrite metadata IDs** (`tools/liveness/appwrite_store.py`):
  database `file_metadata`, collection `production_forecasts`. The
  historical config value `forecasts_metadata` never existed in Appwrite —
  it is the *legacy Postgres schema name* copied into the new store's
  config, and it killed the June 2026 un_fao run (register C-100).
- **Credentials resolution** (`tools/liveness/appwrite_api.py`): process env
  vars first (`APPWRITE_ENDPOINT` / `APPWRITE_DATASTORE_PROJECT_ID` /
  `APPWRITE_DATASTORE_API_KEY`), then known platform `.env` files found by
  ancestor walk. **Secret values are never rendered** — reports show
  `api_key_chars` (a length) only.
- **Appwrite listing** (`tools/liveness/appwrite_api.py`): always
  server-side `orderDesc($createdAt)` + `limit` — client-side sorting of a
  default page is the pagination bug this suite exists to prevent.

## Design rules (hold for any new surface)

TDD — tests first, fixtures are captured real responses
(`tests/test_liveness_*.py`). WET before DRY — shared code
(`report.py`, `appwrite_api.py`) was extracted only after six checks
demonstrably duplicated it. One surface per file; raw facts, no narration;
injected fetch/client/clock seams (DIP); lazy imports in default clients
only; no import-time side effects (C-93); zero new dependencies; truthful
skips (C-75). Unknown verdicts raise `KeyError` in `report.exit_code_for` —
add new verdicts to `EXIT_CODE_BY_VERDICT` deliberately.

## Known non-goals (register C-102 — do not over-read all-green)

The suite does NOT yet watch: **viewser** (the actual input of the four
production ensembles — the datafactory surface covers the input the *next*
system consumes), the **website** (`viewsforecasting.org` is a distinct host
from the API that IS probed), or **content sanity** (delivered file sizes
are reported as facts but not judged — an empty parquet counts as
DELIVERING). Liveness ≠ correctness: green means surfaces are alive and
fresh, not that the numbers in them are right. Closing these is a scope
decision tracked as C-102.

Run the offline suite:

```bash
conda run -n views_pipeline python -m pytest tests/ -k liveness
```
