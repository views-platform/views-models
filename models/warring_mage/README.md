# warring_mage

**DRAFT — views-datafactory `country_month` CM model.** Pending datafactory
CM per-feature aggregation finalization (ADR-040 follow-up).

Based on `smol_cat` (TiDEModel), migrated from viewser to the views-datafactory
`country_month` source with a **minimal UCDP feature set**
(`ged_sb/ns/os_best` → `lr_ged_sb/ns/os`, summed to country level).

- **Target:** `lr_ged_sb`
- **Level:** cm
- **Architecture/hyperparameters:** inherited verbatim from `smol_cat`.

One of 12 CM "label" models intended as the `reconcile_with` target (single or
ensembled) for the reconciling rusty_bucket clone (#144).
