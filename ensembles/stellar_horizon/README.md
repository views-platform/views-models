# stellar_horizon

| Property | Value |
|----------|-------|
| **Algorithm** | HydraNet (HydraBNUNet06_LSTM4) |
| **Level** | pgm |
| **Manager** | PredictionFrameEnsembleManager |
| **Aggregation** | concat (3 x 64 = 192 posterior samples) |
| **Data Source** | views-datafactory |
| **Deployment** | shadow |
| **Creator** | Simon |

## What

Distributional ensemble over three datafactory-powered HydraNet models that differ only in regression loss function:

| Constituent | Loss Function | Key Params |
|-------------|--------------|------------|
| bright_starship | Shrinkage | a=258, c=0.001 |
| bold_comet | Basu DPD | alpha=0.3, sigma=3.0 |
| blazing_meteor | LogNormal NLL | sigma=0.9 |

All three share the same architecture, data source (views-datafactory, africa_me_legacy region), feature set, and partitions. The ensemble concatenates their 64-sample posteriors into a 192-sample joint posterior per target.

## Why

This is the second of three planned HydraNet ensembles, establishing the **datafactory parity baseline**:

1. **golden_hour** (done) -- viewser data, three loss variants. Known-good reference.
2. **stellar_horizon** (this) -- datafactory data, Africa+ME region. Compare parity with golden_hour.
3. **TBD** -- datafactory data, global coverage. Full-scale deployment.

The purpose is to validate that datafactory-based ensembles produce equivalent results to the established viewser pipeline before migrating production workloads.

## Parity Comparison

| Metric | golden_hour (viewser) | stellar_horizon (datafactory) |
|--------|----------------------|------------------------------|
| CRPS lr_sb_best | 0.233 | TBD |
| CRPS lr_os_best | 0.051 | TBD |
| CRPS lr_ns_best | 0.033 | TBD |

## Targets

Regression targets only at ensemble level:
- `lr_sb_best` (state-based fatalities)
- `lr_ns_best` (non-state fatalities)
- `lr_os_best` (one-sided violence fatalities)

Classification targets evaluated at individual model level only (see C-46).

## Metrics

- CRPS (Continuous Ranked Probability Score)
- QS_sample (Quantile Score, sample-based)
- MCR_sample (Mean Calibration Ratio, sample-based)
