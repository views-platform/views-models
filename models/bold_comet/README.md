# bold_comet

| Property | Value |
|----------|-------|
| **Algorithm** | HydraNet (HydraBNUNet06_LSTM4) |
| **Level** | pgm |
| **Data Source** | views-datafactory (not viewser) |
| **Regression Loss** | Basu DPD (alpha=0.3, sigma=3.0) |
| **Deployment** | shadow |
| **Creator** | Simon |

## What

Datafactory-powered HydraNet with Basu DPD regression loss. Cloned from bright_starship with only the loss function changed. Part of the datafactory trio for parity validation against viewser models.

## Parity Mapping

| Datafactory (this trio) | Viewser (golden_hour trio) | Loss |
|------------------------|--------------------------|------|
| bright_starship | purple_alien | Shrinkage (a=258, c=0.001) |
| **bold_comet** (this) | blue_stranger | **Basu DPD (alpha=0.3, sigma=3.0)** |
| blazing_meteor | violet_visitor | LogNormal NLL (sigma=0.9) |

Ensemble: stellar_horizon (datafactory) vs golden_hour (viewser).

## Targets

- `lr_sb_best` (state-based fatalities)
- `lr_ns_best` (non-state fatalities)
- `lr_os_best` (one-sided violence fatalities)
- `by_sb_best`, `by_ns_best`, `by_os_best` (binary classification)

## Prerequisites

Same as bright_starship — requires `views-datafactory` and `~/.netrc` Hetzner credentials. See bright_starship README for setup.

## Usage

```bash
./run.sh -r calibration -t -e
```
