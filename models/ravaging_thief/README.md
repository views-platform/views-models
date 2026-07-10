# ravaging_thief

**views-datafactory `country_month` CM model** — finalized 2026-07-06 against
datafactory ADR-048 (declared feature_agg_types) / v1.6.2. Minimal UCDP count
features **by design** (intensive features are unsafe at CM over the remote
zarr — see risk register C-94/C-95).

Based on `revolving_door` (NHiTSModel), migrated from viewser to the views-datafactory
`country_month` source with a **minimal UCDP feature set**
(`ged_sb/ns/os_best` → `lr_ged_sb/ns/os`, summed to country level).

- **Target:** `lr_ged_os`
- **Level:** cm
- **Architecture/hyperparameters:** inherited verbatim from `revolving_door`.

One of 12 CM "label" models intended as the `reconcile_with` target (single or
ensembled) for the reconciling rusty_bucket clone (#144).
