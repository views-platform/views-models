"""Run one model end to end on a rented GPU, from a bare image to a downloadable deliverable.

**Version 0.1.0 — PROVISIONAL. Not part of the monthly production run.**

This group exists because on 2026-09-24 the operator lost access to fimbulthul and the platform
had to run somewhere we do not control. It was written during that first deployment, and it has
run exactly one campaign: eight HydraNets, calibration partition, global land
(`reports/postmortem_runpod_first_deployment_2026-09.md`).

Read that status honestly before depending on it:

- It has been exercised on **one model family** (HydraNet) and **one partition** (calibration).
  It has never run a stepshifter, an r2darts2, a baseline, an ensemble, or a forecasting
  partition.
- It is **not wired into any CI job, any monthly-run procedure, or `run.sh`**. Nothing calls it
  but a person following `docs/runpod_run_guide.md`.
- It performs **no upload**. It deliberately knows nothing about Appwrite or any publishing
  credential, because it is designed to run on hardware we do not own.

What it does do is refuse early. Every expensive step is preceded by a check that costs seconds,
because the failure that matters on rented hardware is discovering after seven paid hours that a
credential was missing or a config still held a throwaway value.

    bash tools/podrun/pod_run_model.sh <model_name>

Leaves `/workspace/deliver/<model>/` containing the parquets, the compressed posterior, a
MANIFEST, a STATUS of `OK` or `FAILED:<stage>`, and the full transcript.

Promoting this out of 0.1.0 means: a second model family has run through it, the forecasting
partition has run through it, and either CI exercises it or a maintainer other than its author
has used it unaided.
"""

__version__ = "0.1.0"
