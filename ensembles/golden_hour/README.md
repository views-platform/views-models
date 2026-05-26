# golden_hour

| Property | Value |
|----------|-------|
| **Algorithm** | HydraNet (HydraBNUNet06_LSTM4) |
| **Level** | pgm |
| **Manager** | PredictionFrameEnsembleManager |
| **Aggregation** | concat (3 x 64 = 192 posterior samples) |
| **Deployment** | shadow |
| **Creator** | Simon |

## What

Distributional ensemble over three HydraNet models that differ only in regression loss function:

| Constituent | Loss Function | Key Params |
|-------------|--------------|------------|
| purple_alien | Shrinkage | a=258, c=0.001 |
| blue_stranger | Basu DPD | alpha=0.3, sigma=3.0 |
| violet_visitor | LogNormal NLL | sigma=0.9 |

All three share the same architecture, data source (viewser), feature set (lr_sb_best, lr_ns_best, lr_os_best), and partitions. The ensemble concatenates their 64-sample posteriors into a 192-sample joint posterior per target.

## Why

This is the first of three planned HydraNet ensembles, establishing the **viewser data baseline**:

1. **golden_hour** (this) -- viewser data, three loss variants. Known-good reference.
2. **stellar_horizon** -- datafactory data, Africa+ME region. Parity counterpart to golden_hour.
3. **TBD** -- datafactory data, global coverage. Full-scale deployment.

The purpose is to validate that datafactory-based ensembles produce equivalent results to the established viewser pipeline before migrating production workloads.

## Targets

Regression targets only at ensemble level:
- `lr_sb_best` (state-based fatalities)
- `lr_ns_best` (non-state fatalities)
- `lr_os_best` (one-sided violence fatalities)

Classification targets (`by_sb_best`, `by_ns_best`, `by_os_best`) are evaluated at the individual model level but not at ensemble level. See risk register for details on the actuals-derivation gap in PredictionFrameEnsembleManager.

## Metrics

- CRPS (Continuous Ranked Probability Score)
- QS_sample (Quantile Score, sample-based)
- MCR_sample (Mean Calibration Ratio, sample-based)

## Repository Structure

```
golden_hour/
├── main.py
├── run.sh
├── requirements.txt
├── README.md
├── configs/
│   ├── config_meta.py
│   ├── config_hyperparameters.py
│   ├── config_partitions.py
│   └── config_deployment.py
├── data/
│   ├── generated/
│   └── processed/
├── artifacts/
├── logs/
├── notebooks/
└── reports/
```
