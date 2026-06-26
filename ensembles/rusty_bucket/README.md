# Rusty Bucket
## Overview

This folder contains code for the Rusty Bucket model — the **FAO forecast delivery ensemble**. It pools the full posterior draws of its constituent models (`aggregation="concat"`) and ships them as a real distribution, never a collapsed point estimate. The single principled MAP/HDI collapse happens once, downstream, in the summarizer (views-frames#89).

**Interim status:** the eight `temporary_*` constituents are stand-ins — clones of the `heavy_strider` global-land datafactory baseline — used until the real ~8 global HydraNet constituents finish tuning (#146). Eight identical clones make a degenerate mixture by design; the purpose now is to validate the pooled-draw machinery (8-model concat, the equal-samples gate, the no-collapse delivery path, the `un_fao` wiring) at the correct global-land shape. Statistical diversity arrives with the real models.

| Information         | Details                        |
|---------------------|--------------------------------|
| **Models** | temporary_otter, temporary_robin, temporary_finch, temporary_heron, temporary_lynx, temporary_bison, temporary_crane, temporary_fox |
| **Level of Analysis** | pgm            |
| **Targets**         | lr_sb_best, lr_ns_best, lr_os_best |
| **Aggregation**       |  concat (pooled draws, no point collapse)   |
| **Samples**       |  128 per constituent → 1024 pooled (concat concatenates the sample axis: 8 × 128; ADR-015)    |
| **Deployment Status**       |  shadow    |

## Repository Structure

```
Rusty Bucket
├── README.md
├── main.py
├── requirements.txt
├── run.sh
├── logs
├── artifacts
├── configs
│   ├── config_deployment.py
│   ├── config_hyperparameters.py
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
