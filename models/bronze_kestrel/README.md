# Bronze Kestrel 
## Overview


| Information         | Details                        |
|---------------------|--------------------------------|
| **Model Algorithm** | HydraNet                  |
| **Level of Analysis** | pgm            |
| **Targets**         | lr_sb_best, lr_ns_best, lr_os_best |
| **Features**       |  bronze_kestrel_features   |
| **Feature Description**       |  No description provided    |
| **Metrics**       |  No information provided    |
| **Maturity**       |  candidate    |
| **Data Source**       |  datafactory    |

## Repository Structure

```
Bronze Kestrel
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
│   ├── config_partitions.py
│   ├── config_queryset.py
│   ├── config_sweep.py
├── data
│   ├── generated
│   ├── processed
│   ├── raw
├── reports
├── notebooks
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
<!--
  These markers are load-bearing, not decoration. `tools/catalogs/update_readme.py`
  rebuilds EVERY model README from a scaffold and discards anything outside a manual
  block; `.github/workflows/update_catalogs.yml` runs it on every push to development
  that touches models/*/configs/config_*.py and commits the result back. Without these
  markers everything below would be deleted by CI on the merge that adds it, which is
  what happened to synthetic_chant's evaluation-semantics docs on 2026-06-04 (C-77/C-78).
-->

One of **three seed replicates** of `amber_kestrel`. Not a roster member, not an ensemble
constituent, not an experiment arm. Maturity `candidate`.

## What it is for

It exists to measure **one number**: `σ̂`, the run-to-run variance of this training pipeline
at fixed config. Epic **views-models#565**, story **#567**, pre-registered by the
views-hydranet seat in `reports/2026-10-18_weight_decay_arms_prereg.md`.

**Why that number does not exist yet.** Every gate result on record in this programme is
**one run per configuration**. So every statement of the form *"arm X's slope differs from
control Y's"* — including the `amber`↔`copper` slope gap of **0.17**, which has been reasoned
from for days — was made with no estimate of how much this pipeline varies run to run at
fixed config. `σ̂` is the denominator all of them were missing, and it sizes every future
experiment on this architecture.

| replicate | `torch_seed` / `np_seed` |
|---|---|
| `silver_kestrel` | 42 |
| `bronze_kestrel` | 43 |
| `cobalt_kestrel` | 44 |

## ⛔ It is not a contrast, and both ways of breaking that are different

The only permitted difference from `amber_kestrel` is the **seed**, plus instrumentation
(`trajectory_log_path`). The two failure directions are **not** symmetric:

| corruption | effect on `σ̂` | what it looks like |
|---|---|---|
| a config diverges from the others | **inflates** it | thresholds rise, effects fail — mimics the null we already expect. **Conservative.** |
| two replicates share a seed | **deflates** it | thresholds → 0, anything clears — mimics a **resolved effect**. ⛔ **Anti-conservative.** |

Every threshold in the pre-registration is `C = k·σ̂`, so `σ̂` is in the **denominator**.
Identical seeds can only make runs *more* alike than distinct seeds would, so `σ̂` can only
be under-stated and a threshold divided by an under-stated spread can only be too
permissive. ⚠️ **That argument needs no determinism assumption**, which matters because
Amendment 6 withdrew determinism as a precondition of this design.

Guarded by `test_the_replicates_are_distinct_runs` and by the pre-registration's **F13
clause (c)**.

## ⛔ `amber_kestrel` is the config reference, not a member of the cell

It was trained on a **different pod session at a different time under a different code sha**,
so pooling it would import every difference between then and now into the variance estimate
(prereg §7; and Simon, 2026-10-18 — *"I agree with the fresh run, don't reuse amber"*).

**This is why `silver_kestrel` carrying amber's own seed 42 is correct** and not a duplicate
run. The cell is the three replicates; amber supplies the configuration and the published
`b = 1.1875` they are comparable to.

## Do not

- **Do not** vary anything but the seed to "improve" the result — it stops being a replicate.
- **Do not** tidy the inherited docstrings in `config_queryset.py` or `config_sweep.py`.
  They are compared **byte-for-byte** against the control with the model name normalised
  out; a typo fix or a reflow in the copy alone turns the guard red. The stale prose is
  correct *as a copy*.
- **Do not** edit `run.sh`. It is generated production infrastructure, asserted here and
  never edited.
<!-- /manual -->
