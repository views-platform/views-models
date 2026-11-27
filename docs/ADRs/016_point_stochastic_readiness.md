# ADR-016: Point/Stochastic Discriminator for PredictionFrame Readiness

**Status:** Accepted
**Date:** 2026-06-27
**Deciders:** Simon (project maintainer)
**Informed:** All contributors
**Related ADRs:** [ADR-015](015_posterior_sample_count_standard.md) (the 128 sample-count standard — governs *stochastic* models; this ADR carves out the *point* case), [ADR-013](013_regression_target_name_agnosticism.md) (config is the single source of truth — derive, never hardcode); views-pipeline-core **#159** (universal-PF content descriptor); HydraNet `evaluation_mode`.

---

## Context

`PredictionFrame` is becoming the universal forecast container across the platform (views-pipeline-core #159) — including **deterministic / point** models that do no sampling. But the views-models production-readiness contract (`tests/test_pfe_production_readiness.py`) conflated "is a PredictionFrame" with "is sampled": `_discover_pf_models()` selects every model with `config_meta.prediction_format == "prediction_frame"`, and `TestPFModelConfigReadiness.test_has_n_posterior_samples` then demanded a positive-int `n_posterior_samples` from **all** of them.

The six point baselines (`{zero,locf,average}_{cm,pgm}baseline`, algorithms `ZeroModel`/`LocfModel`/`AverageModel`) and three point synthetics (`{diagonal,horizontal,vertical}_dream`) — which return `(N, 1)` and do no sampling — could satisfy the contract only by declaring a **meaningless `n_posterior_samples: 1`** (introduced by commit `3ff5647`). A `ZeroModel` claiming it draws one sample is dishonest: it misleads every consumer that reads sample count to infer "this frame is sampled," and it is not forward-compatible with #159.

---

## Decision

A PredictionFrame model declares its **content mode** via `config_meta.evaluation_mode`:

| Aspect | Rule |
|---|---|
| **Field / location** | `evaluation_mode` in `config_meta` (a model-identity property, beside `prediction_format`). |
| **Values** | `"point"` \| `"stochastic"`. `"parametric"` is reserved (see #159) but not implemented. |
| **Default** | A missing `evaluation_mode` resolves to `"stochastic"` — back-compatible; only point models must opt in. |
| **Point models** | Do no sampling: they **omit** `n_posterior_samples` entirely, and emit `y_pred` of shape `(N, 1)`. Declaring any `n_posterior_samples` (including `1`) is **incoherent** and rejected. |
| **Stochastic models** | Declare a positive-int `n_posterior_samples` (the ADR-015 standard is 128) and emit `(N, n_posterior_samples)`. |
| **Ensemble aggregation** | A point constituent contributes **1** column to the pooled sample axis, so `concat` (sum) and `arithmetic_mean` expectations stay correct for ensembles mixing point + stochastic constituents. |

---

## Rationale

- **Honesty over a workaround.** `n_posterior_samples: 1` on a non-sampling model encodes a falsehood. An explicit mode lets a model say what it is.
- **Reuse, not reinvent.** `evaluation_mode: point|stochastic` is HydraNet's existing production vocabulary (`views_hydranet/utils/config_initializer.py:194`; `collapse_to_point()` → `(N, 1)`), so the platform shares one term.
- **Forward-compatible.** #159 generalizes this to a self-describing PF content descriptor (extensible to `parametric`). Adopting the vocabulary now means the canonical form can adopt or adapt views-models' choice rather than collide with it.
- **`config_meta`, not `config_hyperparameters`.** Mode is identity (what the model *is*), not a tuning knob; it sits with `prediction_format`/`algorithm`/`level`.

---

## Considered Alternatives

### Alternative A: require `n_posterior_samples == 1` for point models
- **Pros:** uniform "every PF model has a positive-int sample count" invariant; smallest test change.
- **Cons:** re-encodes the exact dishonesty this ADR removes; a `1` is indistinguishable from "one sample of a degenerate distribution."
- **Reason for rejection:** semantic honesty is the whole point.

### Alternative B: infer mode from `y_pred.shape[1]`
- **Pros:** no new config key.
- **Cons:** shape is ambiguous (`1` ⇒ point, or one sample of many?; a future parametric frame breaks it entirely), and config-level (pre-run) checks have no array to inspect.
- **Reason for rejection:** #159's core argument — shape alone cannot carry semantics.

### Alternative C: wait for pipeline-core #159 to canonicalize the descriptor
- **Pros:** one canonical form, no later migration.
- **Cons:** #159 is open and explicitly leaves field name/location to consumers, inviting them to move locally now.
- **Reason for rejection:** blocks an in-repo correctness fix on an open cross-repo design; the local choice is coordinated on #159 for alignment.

---

## Consequences

### Positive
- Point models describe themselves honestly; consumers branch on an explicit field, not a guessed shape.
- The readiness contract validates each model *as what it is*; the fake `n_posterior_samples: 1` is gone from all nine point configs.
- Forward path to `parametric` content is open without repainting.

### Negative
- A new config key to understand and to set on future point models (a point model that forgets it defaults to stochastic and **fails loud** on the missing sample count — intended).
- If #159 later canonicalizes a different name or a frame-level location, views-models will need a follow-up alignment.

---

## Implementation Notes

`tests/test_pfe_production_readiness.py`:
- `_model_eval_mode(name)` — resolves the mode from `config_meta` (default `stochastic`).
- `_n_posterior_samples_ok(mode, n)` — pure config-level predicate (point ⇒ `n is None`; stochastic ⇒ positive int).
- `_expected_output_width(name)` — point ⇒ 1; stochastic ⇒ `n_posterior_samples`.
- `_constituent_sample_count(model, ensemble)` — point constituent ⇒ 1 column.

The nine point configs declare `"evaluation_mode": "point"` in `config_meta.py` and carry no `n_posterior_samples`.

---

## Validation & Monitoring

`TestPointStochasticReadinessContract` (story #221) locks the contract: point-must-omit, stochastic-requires-positive-int, missing-defaults-stochastic, point-contributes-one-column, and a positive check that the real shipped point models pass without a sample count (which fails loud if anyone re-adds `n_posterior_samples` to them).

---

## Open Questions

- The `parametric` content case (named parameters + distribution family) — deferred to #159.
- Whether the canonical descriptor ultimately lives on the `PredictionFrame` itself (self-describing) rather than only in config — #159's call.

---

## References

- Epic **#216** (point-aware PF readiness); stories **#217**–**#222**; tracking **#223**.
- **#81** — original issue, superseded by #216.
- views-pipeline-core **#159** — universal-PF content descriptor (coordination).
- views-baseline **PR #15** (merged) — point models return `PredictionFrame`.
- Commit `3ff5647` — introduced the `n_posterior_samples: 1` workaround now removed.
- [ADR-015](015_posterior_sample_count_standard.md), [ADR-013](013_regression_target_name_agnosticism.md).
