# Indigo Kestrel 
## Overview


| Information         | Details                        |
|---------------------|--------------------------------|
| **Model Algorithm** | HydraNet                  |
| **Level of Analysis** | pgm            |
| **Targets**         | lr_sb_best, lr_ns_best, lr_os_best |
| **Features**       |  indigo_kestrel_features   |
| **Feature Description**       |  No description provided    |
| **Metrics**       |  No information provided    |
| **Maturity**       |  candidate    |
| **Data Source**       |  datafactory    |

## Repository Structure

```
Indigo Kestrel
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

One of the **two intervention arms** of the weight-decay experiment. Not a roster member,
not an ensemble constituent, not a seed replicate. Maturity `candidate`.

Epic **views-models#565**, story **#568**, pre-registered by the views-hydranet seat in
`reports/2026-10-18_weight_decay_arms_prereg.md` §3.

## The one variable

| | |
|---|---|
| varies | `weight_decay`, **0.1 → 0.0** |
| control | **`silver_kestrel`** |
| seed | **42** — the same as the control's |

## ⛔ Why the control is `silver_kestrel` and not `amber_kestrel`

Because `indigo_kestrel` and `silver_kestrel` are **both seed 42**, so they differ on
`weight_decay` **alone**.

Declared against `amber_kestrel` instead, this arm would differ on **three** keys —
`weight_decay`, `torch_seed`, `np_seed` — which `tests/test_roster_conformance.py:66-70`
names as the confounded case: *"NOT a one-variable pair: the seeds differ too, so any
difference between the two runs is confounded with seed."*

⚠️ **The seed match is the design, not a coincidence to be tidied.**

## What it tests

`weight_decay: 0.1` is **ten times** the AdamW default, and it reaches the three gate heads.
The exemption mechanism exists — `views-hydranet .../train/training_engine.py:113,136` add
`weight_decay: 0.0` groups for `sigma_params` and `log_var_params` — and the heads were
never added to it. The heads also carry **biases**
(`HydraBNrecurrentUnet_06_LSTM4.py:251,289,339`, no `bias=False`, unlike the encoder at
`:201,:207,:212`), so decay shrinks both the weight (→ slope) and the bias (→ level).

Removing it entirely asks what that was buying.

## ⛔ This is not a causal test

The pre-registration's **F12** fires on any sentence saying `weight_decay` is or is not the
cause of the gate's under-dispersion. The programme's honest ledger is derivations **10
right / 1 wrong**, causal narratives **0 right / 8 wrong**. **The primary output of these
five pods is `σ̂`, not a mechanism.**

## ⚠️ Accepted risk, stated before the run

**Nothing in the record says this architecture trains with regularisation removed
entirely.** Whether it does is epic output **M2**.

⛔ If this run diverges, NaNs, or trips the C-184 seed-bimodal gate, it is **VOID — not
negative** — and the readout must state how many of the two zero-decay runs survived
**before** any slope is read. A void arm is an **absent measurement**, never evidence of no
effect.

`0.0` is reachable with no library change, and that was checked rather than assumed:
`config_initializer.py:150` is `Field(..., ge=0.0)` — `ge`, not `gt`, unlike `learning_rate`
on the line directly above — and the value passes straight to AdamW.

## Do not

- **Do not** vary a second value. `ARM_FREE_KEYS["indigo_kestrel"]` is `frozenset({"weight_decay"})`
  and widening it is what that pin exists to prevent.
- **Do not** re-point the control at `amber_kestrel`. See above.
- **Do not** tidy the inherited docstrings in `config_queryset.py` or `config_sweep.py` —
  they are compared **byte-for-byte** against the control with the model name normalised out.
- **Do not** edit `run.sh`. It is generated production infrastructure.
<!-- /manual -->
