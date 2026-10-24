"""Evidence about a delivery that its structural checks cannot produce.

A delivery can pass every check in the chain and still be worthless. On 2026-09-30 one did: the
manifest was valid, coverage was exactly ``land_gaul``'s 64,742 cells, the findability guard
resolved all 111 objects — and ``tower_point``, the estimator views-faoapi serves, returned zero
for every one of 2,333,448 cells. The posterior was empty. Nothing in the chain said so, and it
was found by hand at 4am by someone computing anchor values for an unrelated probe.

This package produces the two things that would have said so, on every run:

    python -m tools.prereg.posterior_health <pooled_dir> --mode rehearsal|production
    python -m tools.prereg.capture_anchors  <pooled_dir> --mode ... --out P8_ANCHORS.json

``posterior_health`` answers *is there anything in it?* — the fraction of cells with no signal in
any draw, the largest draw, and how many cells carry a non-zero served point estimate. The same
numbers mean opposite things for a rehearsal and a production run, so it is told which it is; it
cannot be inferred from the data, and that is the whole reason a human kept having to.

``capture_anchors`` implements pre-registration v3's **P8** — *the values served ARE the values
we produced*. It records both the point estimate and the raw draws, because the point estimate
alone could not discriminate on the run that motivated this: comparing 0.0 to 0.0 would have
passed against an unrelated empty dataset.

House rules. Both read the pooled frame as the ensemble writes it — ``y_pred.npy`` plus
``identifiers.npz``, not ``PredictionFrame.load``'s ``values.npy``. Anchor selection is seeded, so
it is reproducible and provably not steered by what the API returned. Neither tool can PREVENT
anything: pooling and publishing are a single invocation, so by the time a posterior can be
inspected it is already on the shelf. They exist to make it impossible to miss, not impossible
to happen — the refusal, if one is ever wanted, belongs where the publish decision is made
(views-models#523).
"""
