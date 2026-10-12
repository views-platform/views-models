# amber_kestrel

One arm of the composition experiment that **views-hydranet#396** ends in. Not a roster
member, not an ensemble constituent, maturity `candidate`.

## What it is

`violet_visitor` with three values changed, all of them composition:

| | violet_visitor (control) | amber_kestrel (arm) |
|---|---|---|
| `output_distribution` | `nb` | `truncated_nb` |
| `forecast_composition` | `threshold_gate` | `soft_gate` |
| `gate_threshold` | `0.20` | *absent — raises if present* |

Everything else is copied unchanged: architecture, seeds (42/42), 300 lessons, features,
targets, grid, losses, `loss_class_pos_weight: 2.0`, 4x4 = 16 draws. That is deliberate.
The control for this arm is not a second training run — it is the **2026-10-12
`violet_visitor` vehicle** (`calibration_model_20261012_033112.pt`, sha256
`4acbf4d3c2b457b0...`), which was trained at 300 lessons on global land with its 13
origins of posterior draws. Keeping every non-composition value identical is what makes
that vehicle a valid control and the experiment one arm instead of two.

## Why it is a directory and not an edit

`tools/podrun/pod_run_model.sh` clones `-b development` on the pod. The config a pod
trains is whatever is on `development`, not whatever is in the operator's working tree, so
an arm that exists only locally trains as the control — silently. And editing
`violet_visitor` in place would redden `tests/test_roster_conformance.py`, which pins it
to `("nb", "threshold_gate", 0.20, 42)`, and would make the control un-citable as the
delivered model.

## Before it is run

The pre-registration for the run lives with **views-hydranet#396**, not here. Nothing in
this directory approves a spend.

⛔ The gate AP guard is a **one-sided kill at a large threshold**, sized to the collapses
on record (`C-228`: gate AP 0.31 → 0.13; M45: −0.238). Not "any AP loss": a single pair
against the existing vehicle cannot resolve the ~0.019 seed standard deviation, and
`C-119`/`C-184` record ~20% training variance and BatchNorm seed-bimodality. A guard set
at any-loss would fire on noise.
