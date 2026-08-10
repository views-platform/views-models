# How a forecast actually reaches a consumer — the map

> **This is not an ADR.** It is a description of how the system works *right now*, and it is expected
> to change. When a legacy element retires, its `[LEGACY]` line here retires with it — so this page
> is a **shrinking list, by design**.
>
> That is exactly why it is not an ADR: ADR-000 says decisions are *"never deleted… superseded, not
> erased"*, and a page designed to shrink cannot live under that rule. ADR-017 and ADR-019 cite this
> page instead of containing it.
>
> **The names here are not examples — they are the current state.** `rusty_bucket`, `un_fao`,
> `pink_ponyclub` and the rest are what exists on 2026-08-04. That is the point of this page, and it
> is also why it is not an ADR: when the source feeding a consumer changes, this page changes with it.
> The ADRs describe the shape; this page records what currently occupies it.
>
> **Last re-traced against the code and the live buckets: 2026-08-05.**
> Every claim below names the file or the bucket it came from, so it can be re-checked rather than
> believed.

---

## Why this page exists

Two things make today's picture harder than it should be:

- there are **two stores** — two different central places a forecast can land; and
- the shelf that matters holds **two disjoint dialects** of document.

Neither is a design; both are transitional. This page makes them visible so the ADRs can stand on
solid ground instead of describing an idealised system.

---

## The monthly run — this is what actually ships to the public API `[LEGACY]`

`monthly_run.sh` is a hand-run list. It runs four **legacy DataFrame ensembles** —
`pink_ponyclub`, `skinny_love`, `rude_boy`, `first_love` — each with `-m`.

`-m` is `--monthly`: it bundles train + forecast + report + **prediction_store**. There is no way to
run `monthly_run.sh` as written without publishing.

Each run goes through **one** method — `_save_predictions` → `PredictionIOManager` — which writes the
forecast three ways:

```
monthly ensemble run (-m)  ->  PredictionIOManager                    [LEGACY]
  ├─ local disk: pandas-parquet (legacy list-in-cell)
  ├─ legacy "views-forecasts" store  (df.forecasts.to_store)          [LEGACY]
  │    -> external API  api.viewsforecasting.org                      [MAIN PUBLIC LINE]
  │       (prio-data/views_api, external)
  └─ Appwrite SHELF: production_forecasts,  type="ensemble"           [LEGACY dialect]
```

There *is* a "savers" path — the single-model PredictionFrame path with the composed
`LocalParquetSaver` / `ViewsForecastsSaver` / `AppwriteSaver` trio (`model.py:572`). It is a
**different, conditional** mechanism `[CURRENT]`, and it is **not** what the monthly ensembles use.

## `rusty_bucket` is a different animal `[TARGET]`

`rusty_bucket` is a **PredictionFrame ensemble (PFE)** — the shape everything is converging toward.
It uses neither path above. It:

- writes `save_pf` (npy/npz) to local disk; and
- with the store on, publishes **wire shards** to the *same shelf*, tagged `type="sampled_forecast_*"`.

Those shards are defined by **views-postprocessing's ADR-013**, the Sampled-Forecast Wire Contract.

> *Note on numbering:* on this page **"ADR-013" always means views-postprocessing's ADR-013** — a
> different repo's ADR. It is not this repo's `013_regression_target_name_agnosticism.md`. Written
> **vpp ADR-013** where confusion is likely.

## So the one shelf holds two disjoint dialects

- legacy documents tagged `type="ensemble"`, and
- contract shards tagged `type="sampled_forecast_*"`.

They never overlap. That separation is vpp ADR-013 §11.4's transition invariant.

## The FAO line — one leg, not two

**Corrected 2026-08-04.** This page previously described a *live legacy leg* and a *dormant contract
leg*. That is now inverted: the legacy leg has been **deleted**, and the contract leg is the only one.

```
production_forecasts   type="sampled_forecast_*"
  -> vpp unfao manager: source_selection.resolve_run(...)
       newest FULLY MANIFESTED run for the ensemble named in the launcher config
  -> enrich (GAUL sidecar) -> validate -> unfao_bucket
  -> views-faoapi serves unfao_bucket
```

**How to re-check each step:**

| claim | where |
|---|---|
| the legacy pandas reader is gone | `unfao/managers/unfao.py::_read_forecast_data` — *"ADR-013 contract only… retired in #149"*. `LEGACY_FORECAST_FILTERS` no longer exists in the file. |
| omitting the contract key is a refusal, not a fallback | `contract/launch_config.py::assert_contract_mode` raises unless `wire_contract` is truthy (register C-63) |
| selection is by identity, not recency | `contract/wire/source_selection.py::resolve_run(port, expected_targets, expected_ensemble, …)` |
| faoapi reads `unfao_bucket`, not the shelf | `views-faoapi/src/views_faoapi/managers/api.py:148,278`; `forecast/ingestion/wire_reader.py:3` |
| the upload is interlocked off by default | `unfao/product.py:36` — `UPLOAD_ENABLED = False`, overridable only by the launcher key `wire_upload_enabled` (vpp ADR-013 §11.4) |

*(`managers/appwrite/config.py:39` in faoapi carries `bucket_id = "production_forecasts"` as a
dataclass default. It is overridden at `api.py:278`. It is **not** a second reader — a stale default,
nothing more.)*

## What is actually true today — and it is not what the register says

**On 2026-08-04, the FAO forecast stream is 145 days stale (#320), and a complete forecast has been
sitting on the shelf unshipped since 27 July.**

- `production_forecasts` holds **461 files**, among them exactly one fully-manifested `rusty_bucket`
  run: `rusty_bucket_forecasting_20260727_095355`, with all three `lr_ged_*` target manifests.
- `unfao_bucket`'s newest `forecast_dataset_*` is **2026-03-10**. Its `historical_dataset_*` stream is
  current (5 days) — so the two halves of the FAO delivery have diverged by 140 days.

**Register entry C-97 is marked Resolved (2026-07-28) on the claim that *"the manifest-addressed run
`rusty_bucket_forecasting_20260727_095355` is what faoapi serves."* That is not true.** faoapi reads
`unfao_bucket`; the run is in `production_forecasts`; nothing carried it across. Run-0 produced the
artifact and stopped there.

**The correct statement of the crossed-wires problem is therefore:** the wire is now built on both
sides — the producer emits contract shards, the consumer can ingest them (faoapi #204 closed,
`wire_reader.py` on `main`) — and the step that *moves the file between them* has never run in
anger. It is a delivery that is fully constructed and has not been performed.

That is a different failure from the one this page used to describe, and a more tractable one.

## Two stores, two roles

- **The `views-forecasts` store** (the old one) — pandas-only front door (`df.forecasts.to_store`);
  feeds the public API. Unstructured, matched by fragile naming — the pile nobody opens. *(It also has
  two **non-delivery** roles no ADR governs — legacy-ensemble constituent transport, and a run-metadata
  registry — whose retirement is sequenced by the pipeline-core roadmap.)*
- **The Appwrite `production_forecasts` shelf** (the new one) — the two-dialect bucket above; feeds
  the FAO postprocessor. vpp ADR-013 makes its contract dialect addressable by *declared provenance*
  instead of filename.

## Direction of travel — what is dying, and where it goes

*Context only. This convergence is owned by the views-frames migration and vpp ADR-013, and sequenced
by pipeline-core's Lean Platform End-State Roadmap (2026-07-27) — **not decided here or by any ADR in
this repo**.*

- **DataFrame ensembles** (the four monthly) → **PredictionFrame / PFE** (the `rusty_bucket` shape).
- **the `views-forecasts` store** → **the Appwrite shelf** (→ Hetzner, eventually).
- **shelf dialect `type="ensemble"`** → **`type="sampled_forecast_*"`** (vpp ADR-013 §11.4).
- **the legacy FAO leg** → **the contract FAO leg.** — **DONE**, retired in #149.

Everything converges on one shape: **frames-native producers → the shelf's contract dialect →
contract-reading consumers.** The ADRs are written for that **target** state. The legacy machinery is
*grandfathered* — described here so it is visible, not endorsed.

---

## Where each config lives today

- **A source's maturity-ish label:** `models/<m>/configs/config_deployment.py` → `deployment_status`.
  *(Measured 2026-08-10: **115 `shadow`, 6 `baseline`, 4 `deprecated`, 1 `deployed`** across 126 files —
  130 source directories exist, so four carry no maturity at all. (Epic #242 S3 retired the two datafactory-
  parity ensembles `golden_hour` + `stellar_horizon`, both `shadow` — hence 117→115.) Both quote styles must be counted;
  see ADR-017 §2 and register C-127.
  The single `deployed` source is `ensembles/white_mustang` — whose two members, `lavender_haze` and
  `blank_space`, are both `shadow`.)*
- **An ensemble's members:** `ensembles/<e>/configs/config_modelset.py`.
- **Which ensembles reconcile, and against what:** `ensembles/<e>/configs/config_meta.py` —
  `"reconciliation"` (the method) and `"reconcile_with"` (the partner). Two ensembles declare it:
  `skinny_love → pink_ponyclub`, `white_mustang → cruel_summer`.
- **The FAO "which source feeds us" declaration:** `deliveries/un_fao.py` — the `send` line.
  `postprocessors/un_fao/configs/config_meta.py` **derives** its `"ensemble"` key from it and no longer
  names a source (#347). The key survives because views-postprocessing reads `configs["ensemble"]` at
  `unfao/managers/unfao.py:195`; the decision moved, the interface did not.
  - *This entry used to describe a smell:* that config's docstring claimed the file was documentation
    only, while one line in it decided which forecast reached the UN. #347 removed the line and rewrote
    the docstring. **Recorded because it is what ADR-019 was written to fix — and because a map that
    keeps reporting a repaired defect teaches readers to distrust it.**
- **Whether the FAO delivery actually uploads:** `intent` in `deliveries/un_fao.py`. The launcher key
  `wire_upload_enabled` is **derived** from it (#348) and is now committed, so arming is answerable
  from a clean checkout — closing the observable half of **C-110**.
  - **Arming is withheld when the repository disagrees with itself.** The delivery declares `coverage`;
    `config_queryset.py` declares `REGION`. If they differ the upload disarms with a warning naming the
    file, rather than shipping a region nobody declared. It warns rather than raising, so a run that
    never intended to upload still works (vpp ADR-013 §11.4 stages artifacts locally).
  - **C-110's residual is now `wire_contract` and `region`**, still working-tree only. A clean checkout
    carries `REGION = "africa_me_legacy"`, so it disarms — visibly, with the file named — instead of
    delivering the wrong region.
- **The main public line's declaration:** *none exists.* It is emergent — `monthly_run.sh`, plus the
  legacy store, plus the external API.

---

## References

- **ADR-017** — sources, composition and delivery (the three axes). Cites this page for today's state.
- **ADR-019** — the delivery declaration (the file format that replaces the buried `"ensemble"` line).
- **ADR-020** — errors must descend.
- **views-postprocessing ADR-013** — the wire contract; owns *how* bytes travel.
- **pipeline-core** *Lean Platform End-State Roadmap* (2026-07-27) — owns the retirement sequencing.
- Register: **C-97** (selection; Resolved on a claim this page contradicts), **C-110** (uncommitted config — residual narrowed to `wire_contract`
  and `region` by #348), **C-121** (no age bound at the delivery boundary), **C-123** (`rusty_bucket`'s config
  does not describe what it emits), **#320** (the 145-day stall).
