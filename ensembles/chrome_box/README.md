# Chrome Box 
## Overview

This folder contains code for the Chrome Box model, an ensemble machine learning model designed for predicting fatalities. 


| Information         | Details                        |
|---------------------|--------------------------------|
| **Models** | ashen_sparrow, brisk_sparrow, candid_sparrow, dusky_sparrow, eager_sparrow, faint_sparrow, gentle_sparrow, humble_sparrow                  |
| **Level of Analysis** | pgm            |
| **Targets**         | lr_sb_best, lr_ns_best, lr_os_best |
| **Aggregation**       |  concat   |
| **Metrics**       |  No information provided    |
| **Maturity**       |  candidate    |

## Repository Structure

```
Chrome Box
├── README.md
├── main.py
├── requirements.txt
├── run.sh
├── logs
├── artifacts
├── configs
│   ├── config_hyperparameters.py
│   ├── config_maturity.py
│   ├── config_meta.py
│   ├── config_modelset.py
│   ├── config_partitions.py
├── data
│   ├── generated
│   ├── processed
├── reports
```

## Setup Instructions

Clone the [views-pipeline-core](https://github.com/views-platform/views-pipeline-core) and the [views-models](https://github.com/views-platform/views-models) repository.


## Usage
Modify configurations in configs/.

If you already have an existing environment, run the `main.py` file. If you don't have an existing environment, run the `run.sh` file. 

```
python main.py -r calibration -t -e

or

./run.sh -r calibration -t -e
```



<!-- manual -->
## Running this ensemble on a fresh box

The same correction as `chrome_crate`: `--saved` is wrong, and the command this ensemble pair used
to document is refused before any member runs (views-models#639).

```
python -m ensembles.chrome_crate.main -r forecasting -t -f
python -m ensembles.chrome_box.main -r forecasting -t -f
```

`-t` because no member has an artifact, `-f` because nothing else forecasts, no `--saved` because
the member training step would then look for local data a fresh box lacks, and no `-p` because
nothing here publishes. `chrome_crate`'s README explains each flag against the code that reads it.

### The order and the filesystem are both load-bearing

This ensemble reconciles against `chrome_crate` (`reconcile_with` in `configs/config_meta.py`) and
reads that ensemble's forecast **from local disk**, not from any store: `load_cm_frame` →
`EnsemblePathManager("chrome_crate")._get_generated_predictions_data_file_paths("forecasting")`.

- `chrome_crate` must have **completed** a forecasting run on the **same filesystem**. If it has
  not, this run fails with *"no local forecast found for country model 'chrome_crate'"*.
- `load_cm_frame` reads `paths[0]`. The scan is `sorted(paths, reverse=True)` over files whose stem
  starts with `predictions_forecasting`, and the run timestamps are fixed-width `YYYYMMDD_HHMMSS`,
  so `paths[0]` is indeed the newest run — **as long as every file in `data/generated/` is
  machine-named.** The match is a `startswith`, so a hand-copied
  `predictions_forecasting_backup.parquet` sorts above every timestamp and would silently become
  the country totals. Do not leave hand-named files in that directory.

### Memory

This ensemble peaks at **~53.9 GiB**, not the ~28.6 GB of the draw pool alone: `_forecast_ensemble`
still holds `model_results` while `forecasts[target]` accumulates, so peak is about twice pool
(views-models#638). Rent the ~87 GiB class — the `RAM >= 50 GB` rule in
`docs/runpod_run_guide.md:98` does not cover this ensemble. That figure is a floor: transients
inside `np.concatenate` and in the reconciler are unmeasured.
<!-- /manual -->
