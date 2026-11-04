# Copper Kestrel 
## Overview


| Information         | Details                        |
|---------------------|--------------------------------|
| **Model Algorithm** | HydraNet                  |
| **Level of Analysis** | pgm            |
| **Targets**         | lr_sb_best, lr_ns_best, lr_os_best |
| **Features**       |  copper_kestrel_features   |
| **Feature Description**       |  No description provided    |
| **Metrics**       |  No information provided    |
| **Maturity**       |  candidate    |
| **Data Source**       |  datafactory    |

## Repository Structure

```
Copper Kestrel
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

One arm of the gate-objective experiment pre-registered in views-hydranet
`reports/2026-10-16_pos_weight_prereg.md`. Not a roster member, not an ensemble
constituent, maturity `candidate`.

## What it is

`amber_kestrel` with **one** value changed:

| | amber_kestrel (control) | copper_kestrel (arm) |
|---|---|---|
| `loss_class_pos_weight` | `2.0` | `1.0` |

Everything else is copied unchanged: the composition (`truncated_nb` x `soft_gate`, no
`gate_threshold`), architecture, seeds (42/42), 300 lessons, features, targets, grid,
losses, 4x4 = 16 draws. `body_nll_denominator` is deliberately **not set**, so it keeps the
library default `"positives"` — the denominator fix is a separate pre-registration
(`reports/2026-10-15_loss_balance_prereg.md`) and must not ride along, or neither arm is
one-variable.

The control is not a second training run — it is the **as-run 2026-10-13 `amber_kestrel`
arm** (`calibration_model_20261013_010739.pt`, sha256 `2a5efc2a67c08eac...`, git
`73c3f42`).

## Why it is a directory and not an edit

`tools/podrun/pod_run_model.sh` clones `-b development` on the pod, so the config a pod
trains is whatever is on `development` — an arm that exists only locally trains as the
control, silently.

And editing `amber_kestrel` in place was rejected deliberately (Simon, 2026-10-16):
`amber_kestrel`'s own control artefact cannot be retrained (`C-317`), so varying a second
key there would have turned the `amber_kestrel` ↔ `violet_visitor` composition contrast
into a two-variable one **permanently**, spending a finding we already have to buy one we
have not made yet.

## The chain, because no test asserts it

`copper_kestrel` → `amber_kestrel` → `violet_visitor`. The one-variable guard in
`tests/test_roster_conformance.py` is **non-transitive by design**: each arm is checked
only against its own declared control, because a transitive set would admit the union of
every hop's free keys and weaken by one key per arm. Cumulative distance from
`violet_visitor` is therefore three values plus the absence of `gate_threshold` — a fact
for this README, not a contrast anything measures.

## Before it is run

The pre-registration lives in views-hydranet, not here. Nothing in this directory approves
a spend.

⛔ **This arm pre-registers that the headline occurrence metric does NOT move.** The
weighted-BCE tilt `s* = wp/(1 + (w-1)p)` is strictly monotone in `p`, and average precision
depends on the gate only through ranks — so removing a pure recalibration leaves every
ordering, and therefore gate AP, exactly unchanged. P-3 pins AP at the as-run arm's
**0.5324 ± 0.010**. If AP *does* move, that is a different finding (the altered loss
changing what the shared trunk learned) and must not be reported as confirming this
hypothesis. The primary endpoint is **calibration**, which is where the motivated mechanism
lands.

⚠️ The contrast carries a measured handicap: the as-run arm's gate AP is 0.5324 against
`violet_visitor`'s 0.5631, and the loss imbalance that the denominator evidence blames for
that gap is **still present here**. Thresholds are sized against 0.5324, and an AP recovery
here licenses no claim about the denominator arm.

⚠️ The pre-flight is **not** the denominator arm's. That gate was a loss-magnitude
assertion; this arm changes no reduction, so it would pass vacuously. This arm's pre-flight
is to confirm from the *built* loss that `pos_weight` is 1.0 and nothing else moved.
<!-- /manual -->
