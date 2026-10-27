# Chrome Crate 
## Overview

This folder contains code for the Chrome Crate model, an ensemble machine learning model designed for predicting fatalities. 


| Information         | Details                        |
|---------------------|--------------------------------|
| **Models** | ashen_condor, brisk_condor, candid_condor, dusky_condor, eager_condor, faint_condor, gentle_condor, humble_condor                  |
| **Level of Analysis** | cm            |
| **Targets**         | lr_sb_best, lr_ns_best, lr_os_best |
| **Aggregation**       |  concat   |
| **Metrics**       |  No information provided    |
| **Maturity**       |  candidate    |

## Repository Structure

```
Chrome Crate
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

⛔ **`--saved` is wrong here, and the command this issue used to document does not start at all.**
`python -m ensembles.chrome_crate.main --run_type forecasting --saved` is refused at argument
validation — *"Run type is forecasting but neither --train, --evaluate, nor --sweep flag is set.
Nothing to do..."* — so it never reaches a member. On top of that, none of the eight `*_condor`
members has a trained artifact anywhere, so there would be nothing for `--saved` to reuse
(views-models#639).

The launch for a box that has never run these models:

```
python -m ensembles.chrome_crate.main -r forecasting -t -f
```

- **`-t`** trains the eight members. Without it each member calls
  `resolve_artifact_path("forecasting")`, which raises `FileNotFoundError` when no artifact exists.
- **`-f`** produces the predictions. Without it `_execute_model_tasks` trains and then stops:
  `if ctx.args.forecast:` is the only branch that forecasts.
- **no `--saved`.** `_create_model_args` computes `saved = ctx.args.saved if train else True`, so
  the member *training* step fetches from the datafactory when `--saved` is absent, and the member
  *forecast* step always reuses what training just cached. Passing `--saved` sends the training
  step looking for local data a fresh box does not have.
- **no `-p`.** Nothing here publishes to the prediction store. The delivery is gated separately
  (views-crafdapi#180, views-models#599), and rule 5 of `docs/runpod_run_guide.md` holds: the only
  credential the box needs is the read-only datafactory entry in `~/.netrc`.

This ensemble runs **first**, before `chrome_box`, and on the same machine. `chrome_box`'s README
says why neither of those is a free choice.
<!-- /manual -->
