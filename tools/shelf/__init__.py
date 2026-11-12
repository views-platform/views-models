"""Putting a finished forecast on the production shelf, and nothing else.

The "shelf" is the Appwrite `production_forecasts` store: where this platform's ensembles
publish their pooled forecasts, and the **only** place the delivery postprocessors read them
from. `views_postprocessing`'s CRAF'd and FAO managers both resolve their input through it
(`crafd/managers/crafd.py:222`), so a forecast that is not on the shelf cannot be delivered,
however complete it is on local disk.

⛔ **Why this group exists at all.** Publishing is normally a side effect of a forecast run —
`publish_sampled_forecast` has one caller in pipeline-core, inside `_forecast_ensemble`. That
is the right design for a run that produces and publishes on one machine. It leaves no route
for a forecast produced somewhere it may not hold a publish credential, which is exactly the
pod case: RunPod ground rule 5 keeps publish credentials off rented hardware, and the member
artifacts needed to re-pool are too large to bring home. Hence a tool, on the operator's
machine, that publishes what the pod already pooled.

Everything here writes to an external store, so everything here is **opt-in and refuses by
default**. A tool in this group that cannot say why it believes what it is about to publish
does not belong in it.
"""
