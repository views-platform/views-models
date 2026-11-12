"""Run one model end to end on a rented GPU, from a bare image to a downloadable deliverable.

**Version 0.1.0 — PROVISIONAL. Not part of the monthly production run.**

This group exists because on 2026-09-24 the operator lost access to fimbulthul and the platform
had to run somewhere we do not control. It was written during that first deployment, and it has
run one campaign to completion: eight HydraNets, calibration partition, global land
(`reports/postmortem_runpod_first_deployment_2026-09.md`).

    bash tools/podrun/pod_run_model.sh <model_name>              # HydraNet
    bash tools/podrun/pod_run_fao_delivery.sh --preflight        # the FAO chain
    bash tools/podrun/pod_run_darts_calibration.sh --preflight <model_name>   # r2darts2

Read the status honestly before depending on any of them:

- Exercised on **two model families** — HydraNet, and r2darts2 since epic #532 — and **one
  partition** (calibration), plus the FAO forecasting chain. It has never run a stepshifter, a
  baseline, or an ensemble on its own.
- **The darts runner has never completed a real run.** Its checks are tested (35 of them,
  mutation-verified) and its preflight has been exercised against all eleven target models'
  configs, but no r2darts2 model has produced a prediction at pgm at all — that is #537, and
  until it passes the training leg of that script is unproven.
- It is **not wired into any CI job, any monthly-run procedure, or `run.sh`**. Nothing calls it
  but a person following `docs/runpod_run_guide.md`.
- `pod_run_model.sh` and `pod_run_darts_calibration.sh` perform **no upload**, and install no
  credential that could. `pod_run_fao_delivery.sh` publishes by design and needs nine publish
  variables — which is precisely why the other two must not be written by copying it. See the
  guide's ground rule 5.

What they do is refuse early. Every expensive step is preceded by a check that costs seconds,
because the failure that matters on rented hardware is discovering after seven paid hours that a
credential was missing or a config still held a throwaway value. `--preflight` reports every
problem at once rather than one per attempt.

Bugs found by review and now guarded, worth keeping visible because each was silent:

- the draws archive could be **empty and still report success** — `find … -print0 | tar --null
  -T -` exits 0 and writes a valid 22-byte archive when nothing matches. Counted before,
  verified after.
- two guards read config *text*, which is the defect this repo already shipped once (#501, "the
  guard that was not one"). Both now load and **call** the config.
- the darts runner's `--preflight` wrote `OK` to `STATUS`, the file an orchestrator reads to
  decide whether output may be used. It writes `PREFLIGHT_OK`; only a finished run writes `OK`.
- the darts runner's per-model lock did not cover `$VENV`, which every model on the pod shares,
  so two models started together on a fresh pod would both build it. The build is serialised,
  and the EXIT trap releases that lock only when the process holds it.

Known and not fixed: `pod_run_model.sh` and `pod_run_fao_delivery.sh` have no automated tests,
and their MANIFEST provenance fields are unchecked — a missing git sha would ship blank rather
than refuse. The darts runner has tests; the other two do not.

**Duplication, and the trigger for ending it.** Three scripts now carry their own copy of
`stage()`, `die()`, the lock, the `PODRUN_ROOT` resolution and the tee — roughly 25 lines, three
times. That is the point at which extraction starts to pay, and it is deliberately **not** done
yet: it would mean editing `pod_run_fao_delivery.sh`, which has completed a real delivery to a
partner. **Extract `tools/podrun/_common.sh` when either a fourth script appears, or a change has
to be made identically in all three.** Not "later".

Promoting out of 0.1.0 means: the forecasting partition has run through it for a second family,
a real darts run has completed (#537), and either CI exercises it or a maintainer other than its
author has used it unaided.
"""

__version__ = "0.1.0"
