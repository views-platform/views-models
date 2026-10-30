# ADR-023: Posterior draws collapse by the method the model declares, in count space, outside the run

**Status:** **Accepted** (2026-09-28) — **amended 2026-10-06** (§6 added: the r2darts2
`dataframe` source, where the draws arrive as a Python list in every cell rather than as a numpy
cube, and a second converter reads them — views-models#533, epic #532. §1–§4's reasons are
unchanged and now cover a second source shape; **one rule is added**, §6.4: a converter must not
write over its own input.)
**Date:** 2026-09-28
**Deciders:** Simon, VIEWS platform team
**Related ADRs:** [ADR-012](012_target_scale_and_prefix_convention.md) (target scale and prefix),
[ADR-015](015_posterior_sample_count_standard.md) (D x K, the posterior sample count),
[ADR-016](016_point_stochastic_readiness.md) (point vs stochastic readiness),
[`vmo_021`](021_coverage_is_declared_once.md) (a reconciliation is not a derivation)
**Related, in views-hydranet** (`views-hydranet@32bc509`; prefixed per the README's
cross-repo collision rule — this repo has its own 021): `vhy_021` *Volume Dimension Reduction*,
`vhy_039` *Sequence* (Predict → Align → Wrap → Invert → Collapse)

---

## Context

The eight HydraNets do not predict a number. Each cell gets a **gate** (`P(y>0)`) and a **body**
(count-distribution parameters), and the run draws from them `D x K` times — `D`
`n_posterior_samples` MC-dropout passes, each re-drawn `K` `n_head_samples` times from the
parameters that pass produced (ADR-015). All eight currently sit at `D=4, K=4`, so every cell
carries **16 draws**. The researchers' tooling — `ensemble-updater` — takes one number per cell:
*"Currently it only supports point predictions!"*

**The platform already has this collapse, and the models already declare how to do it.**
`views_hydranet/utils/inference_orchestrator.py` runs `vhy_039`'s five stages, of which the last
two are:

```
# --- 4. INVERT (ADR 039.4) ---
pred_handler = scaler.inverse_transform_volume(pred_handler)      # :175

# --- 5. COLLAPSE (ADR 039.5) ---
if self.config.get("evaluation_mode") == "point":                 # :177
    pred_handler = pred_handler.collapse_to_point(method=self.config["aggregate_method"])
```

`collapse_to_point` (`volume_handler.py:502`) implements exactly two methods — `arithmetic_mean`
and `median` — and raises `NotImplementedError` on anything else. **All eight models declare
`aggregate_method: 'arithmetic_mean'`.**

The draws nevertheless reach disk uncollapsed, for one reason: all eight also declare
`evaluation_mode: 'stochastic'`, so the guard at `:177` is false and stage 5 never runs. That is
deliberate — the posterior is the scientific product — but it means **something downstream must
perform stage 5, and today nothing does.**

Two further facts shaped where that "something" lives:

- **The parquet the pipeline would have written is switched off, on purpose.** All eight carry
  `skip_predictions_delivery: True`. Track B (`to_arrow_table()` → `to_prediction_df()`) allocated
  ~5.5M Python floats for a ~4.8–6.4 GB peak and had no consumer; the flag was turned on
  2026-05-26 and `CoreConfigSniffer` now makes the key mandatory (**C-47**). Re-enabling it to get
  parquets would revive a known OOM to produce a file we would still have to reshape.
- **The draws on disk are already in count space.** Stage 4 precedes stage 5 so that averaging
  happens in counts, not in `log1p` — `feature_scaler.py:199` says so in as many words:
  *"Essential for accurate Arithmetic Mean collapse (ADR 021)"* — meaning its own, `vhy_021`.
  Verified on disk: violet_visitor's saved draws reach 903. **Nothing downstream may apply
  `expm1`.** The inversion is reached through
  a transform registry rather than a literal call, so grepping for `expm1` in the model repo finds
  nothing and invites exactly the wrong conclusion. It was drawn once in this project already.

What is left on disk is Track A+: `predictions_<run_type>_<ts>/origin_i/<target>/{y_pred.npy,
identifiers.npz}` — 13 rolling origins, each a 36-month horizon one month apart, `lr_*` (the full
gate x body expectation) beside `by_*` (single-cause decompositions).

---

## Decision

### 1. The collapse happens in views-models, from the saved numpy, on the operator's machine.

`tools/collapse/` converts Track A+ numpy into the researchers' parquet. Not in views-hydranet
(the posterior is what that repo is for), not by re-enabling Track B (C-47), and not on the
rented GPU — the numpy comes down first, and the conversion is re-runnable from it forever.

### 2. The method is the one the model declares. It is never hard-coded.

`collapse_origin(..., aggregate_method=...)` implements `arithmetic_mean` and `median` — the same
two names `vhy_021` defines — and **refuses any other string** rather than falling
back to a default. The converter's default is `arithmetic_mean` because all eight declare it.

**That default and the models' declaration are two places stating one fact, so they are pinned
together.** `tests/test_roster_conformance.py::test_collapse_declaration_matches_the_converter`
imports `DEFAULT_AGGREGATE_METHOD` and asserts every roster member matches it, and that every
member is still `stochastic`. A model moving to `median` turns that test red and names the flag to
pass. This is the belt-and-braces assertion `vmo_021` §4 permits, not a reconciliation: there is no
second copy to drift, only a declaration and a check that we honour it.

### 3. The collapse is arithmetic, in count space, and nothing else happens here.

The converter averages the draw axis in `float64` and writes the result. It does **not** apply
`expm1`, re-gate, rescale, reweight, filter cells, or reorder rows. If the numbers arriving are
wrong, the fix is upstream; a converter that could repair them could also disguise them.

### 4. The converter refuses rather than guesses.

Every one of these raises `CollapseError` and writes nothing:

| condition | why refusing beats proceeding |
|---|---|
| a target directory or `y_pred.npy` / `identifiers.npz` missing | a silently absent target ships a parquet short a column |
| targets not row-aligned, or of different length | joining misaligned targets moves numbers between cells |
| targets disagreeing on draw count | they are not from one run |
| identifiers not matching the prediction row count | a truncated `identifiers.npz` relabels every cell after the truncation |
| any non-finite value | a NaN averaged away becomes a plausible number |
| any negative value | counts are non-negative; negatives mean this is not the array we think |
| fewer than 2 draws | already collapsed upstream; averaging again hides that it happened |
| **any one target's** largest value below `MIN_PLAUSIBLE_MAX` (12.0) | that target looks like `log1p` space — **investigate upstream, do not `expm1` here**. Checked per target: one target can be left in log space while its siblings are fine, and a combined maximum is then carried over the threshold by a healthy sibling |
| a repeated `(month_id, priogrid_id)` pair | `ensemble-updater` joins on that pair, so a duplicate silently wins or loses the join. Every target agreeing on a duplicated identifier is still a duplicate, so the row-alignment check cannot see it |

Only `lr_*` targets are read. `by_*` sits in the same directory and is not the deliverable.

### 5. The output shape is fixed by the specification, not by the converter's convenience.

One parquet per origin, `predictions_<run_type>_<ts>_<NN>.parquet`, `NN` = `00`–`12` **numeric**,
columns `month_id`, `priogrid_id`, `pred_lr_sb_best`, `pred_lr_ns_best`, `pred_lr_os_best`.
Origins are ordered by integer index — `origin_10` sorts before `origin_2` as text, and a fixture
of three origins cannot detect that.

### 6. The same decision applies to the r2darts2 `dataframe` source, through a second converter.

*Added 2026-10-06 (#533). §1–§5 were written from the HydraNet source alone. The 11 pgm r2darts2
models (epic #532) reach the same deliverable from a different shape, and the reasons above hold
without change — so this is a second **instance**, not a second decision.*

**6.1 What arrives.** A `prediction_format: "dataframe"` model writes the origins as *files*, not
directories: `predictions_<run_type>_<ts>_<NN>.parquet`, already one per rolling origin, keyed by a
`(month_id, priogrid_id)` **index**, with every cell holding a **Python list** of sample values —
"length 1 for deterministic, length S for probabilistic"
(`views_r2darts2/transformers/darts_bridge.py::prediction_frames_to_dataframe`).

**6.2 Why a second converter rather than a flag on the first.** The two inputs share no format, no
target naming and no draw-count contract. §4's "fewer than 2 draws → refuse" is *correct* for a
posterior cube and *false* for nine of the eleven darts models, which carry exactly one sample by
design. A flag would make that refusal conditional, which is how a guard stops meaning anything.
`tools/collapse/collapse_darts_predictions.py` is the sibling; `collapse_predictions.py` is
untouched, and `tools/collapse/__init__.py` states which reads which.

**6.3 The collapse, restated for a list cell.** Length 1 is **unwrapped**, not averaged — a
deterministic model's single value is the value, and calling it a mean would assert a posterior
that does not exist. Length S > 1 is the arithmetic mean in `float64`, in count space, exactly as
§3 requires. The keys become flat `int64` columns, per the specification's "flat column, not an
index".

**This matters more here than for HydraNet, because the failure is silent.** The consumer does not
collapse: `_as_float_prediction_array` in `ensemble-updater` takes `float(x[0])` on a list cell —
**one draw, silently**. So an unconverted hand-over of darts output does not raise; it publishes
draw zero as the answer. For a deterministic model that is accidentally correct, which is worse,
because it means the mistake only surfaces on the models where it does damage.

**6.4 A converter must not write over its own input.** New rule, and specific to this source: the
HydraNet converter reads a *directory* and writes *files*, so the two cannot collide. Here the
source and the deliverable share one filename pattern, and an in-place default would destroy the
run output that cost the GPU time. The destination is a distinct directory
(`delivery_<run_type>_<ts>/` by default) and any collision is refused.

**6.5 The refusals of §4 that carry over, and the two that change.**

| condition | darts converter |
|---|---|
| non-finite, negative, duplicate `(month_id, priogrid_id)`, per-target `MIN_PLAUSIBLE_MAX` | **unchanged**, same reasons |
| fewer than 2 draws | **dropped** — one sample is the declared configuration of nine of the eleven |
| targets not row-aligned / disagreeing on draw count | **not applicable** — all targets share one frame, so there is no join to misalign |
| *new:* cells of differing length within a column | the sample count varies row to row; no single collapse reconciles that |
| *new:* a gap in the `_00.._NN` sequence, or a non-contiguous set | `ensemble-updater` raises `FileNotFoundError` naming a missing origin, so a gap must not be converted quietly |
| *new:* no `pred_*` column | the metric frames and the run log live in the same directory |

**6.6 What is not pinned, and deliberately.** The target set is **read off the file**, not
hard-coded as §2's `TARGETS` is, because the darts models declare canonical `lr_ged_*`
(views-models#151) and a future model may declare others; and `MIN_PLAUSIBLE_MAX` is **inherited
from §4 and has never been measured against a darts run** — no r2darts2 model has produced a pgm
prediction at all (epic #488's definition-of-done line 4). The refusal message says so, so that an
operator who trips it knows the threshold is a candidate and not only the data.

---

## Rationale

**Why the mean and not something cleverer.** The mean of the draws is the Monte-Carlo estimate of
`E[y]`, which is what "expected fatalities" means and what the models already declare. It is also
what the pipeline itself would have applied at stage 5. Choosing anything else here would make the
delivered file disagree with the model's own configuration while looking identical.

This is worth stating because a better point estimate may well exist. views-hydranet#337 reports
`gate x mu` beating a 16-draw mean substantially on this roster — a variance effect, not a bias
correction. **It is out of scope here and deliberately so:** `body_mean_dump_dir` is a constructor
argument with no config key, so obtaining `gate x mu` means replacing `main.py`, and the
researchers need files now. Changing the estimator is a modelling decision for views-hydranet,
made once, for everyone — not something a delivery script decides on its own.

**Why float64.** The draws are stored `float32`; summing in `float32` makes a cell's value depend
on how many draws were taken. The difference is ~1e-6 on counts and matters to nobody, but the
accumulator costs nothing and makes the number reproducible from the stored array alone. This is a
knowing, documented divergence from `collapse_to_point`, which uses numpy's default accumulator.

---

## Considered Alternatives

### A: Set `skip_predictions_delivery: False` and take the pipeline's parquet
- **Pros:** no new code; the pipeline's own output.
- **Cons:** revives the allocation C-47 exists to prevent; produces list-in-cell parquet that still
  needs collapsing; `_as_float_prediction_array` in `ensemble-updater` takes `float(x[0])` on a
  list cell — **one draw, silently**, which is the exact failure this ADR is meant to prevent.
- **Rejected:** it reverses a decision the maintainer took on measured evidence, to obtain a file
  that is further from the deliverable than the numpy already on disk.

### B: Set `evaluation_mode: 'point'` so the run collapses at stage 5
- **Pros:** uses the platform's own collapse; no new code.
- **Cons:** throws the posterior away at source. The draws are the scientific product and the
  reason for the run; a delivery convenience must not destroy them.
- **Rejected:** collapsing is cheap and repeatable from the numpy; re-running the model is not.

### C: Collapse on the GPU pod before download
- **Pros:** ~10x less to transfer.
- **Cons:** the irreversible step happens on rented hardware that will be destroyed, with no way
  to re-derive if the choice was wrong.
- **Rejected:** bandwidth is cheaper than a re-run.

### D: Hard-code the arithmetic mean
- **Pros:** simplest possible converter; true for all eight today.
- **Cons:** the models *declare* `aggregate_method`. Hard-coding it puts the estimator in two
  places with nothing to reconcile them — the pattern `vmo_021` exists to stop.
- **Rejected after being written this way first.** The first draft hard-coded it; discovering
  `collapse_to_point` is what showed the declaration already existed.

---

## Consequences

### Positive

- The delivered number is the estimator the model declares, and a test fails if that stops being
  true.
- The conversion is re-runnable from the preserved numpy, offline, forever. A mistake in the
  parquet costs seconds, not a GPU run.
- The eight failure modes in §4 are loud. The one that motivated all of them —
  a `log1p`-scaled field delivered as counts — scores as plausible nonsense and is unrecoverable
  once a researcher has acted on it.

### Negative

- **A third place now knows the layout of `predictions_*/origin_i/<target>/`.** views-hydranet
  writes it, pipeline-core reads it, and views-models now parses it. That path is not a published
  contract, and a change to it breaks this converter. The mitigation is loudness, not prevention:
  every structural assumption raises with the offending path named.
- Delivery takes a manual step on the operator's machine. Deliberate — §1 — but it is a step that
  can be forgotten, and the parquets are not produced by the run that produced the numpy.

### Deliberately not done

`gate x mu` (views-hydranet#337) and the D/K split (ADR-015). Both are modelling decisions. This
ADR governs the arithmetic of delivery and must not become the place where the estimator is
chosen by whoever last edited a script.

---

## Implementation Notes

```bash
python -m tools.collapse.collapse_predictions models/<model> --run-type calibration
python -m tools.collapse.plot_collapse_audit <parquet> --draws-dir <origin_i> --out audit.png
```

`--aggregate-method` exists and must be passed if a model ever stops declaring `arithmetic_mean`;
the roster test names it when that happens.

**The plot script is part of the procedure, not a debugging aid.** The tests prove the arithmetic;
they cannot tell you the field stopped looking like conflict. A person looks at the panels before
anything is sent.

---

## Validation & Monitoring

- 31 tests across `tests/test_collapse_predictions.py` (synthetic, contract + input mutation) and
  `tests/test_collapse_on_real_predictions.py` (real output; skips on a clean checkout).
- **23 mutations applied to the converter itself, 23 caught, 0 survived** (2026-09-28). The first
  pass caught 19 of 21; the two survivors — lexicographic origin ordering, and a dropped
  identifier-length check — were coverage holes, and closing them is why the fixture now carries
  13 origins rather than 3.
- **Code review found a guard that could not fire**, and it is the one that mattered most. The
  scale check originally took the maximum of the three targets *flattened together*, so a single
  target left in `log1p` space was carried over the threshold by a healthy sibling and shipped as
  `log1p(count)`. Its test could not detect this either: it set all three targets to log-space
  values at once, so it passed under both the broken and the correct implementation. The check is
  now per target, and a mutation that restores the flattened form is caught.
- Cross-checked against a pure-Python per-row recomputation over all 13 origins of all eight
  models' validation output: **zero mismatches**.
- The converter found two defects in itself under test: a guard that raised `ValueError` while
  building its own error message, and a `float32` accumulator.

**What is not yet validated:** global land, the calibration partition, and `D x K = 32`. All eight
measurements above are Africa+ME validation at 16 draws, because that is what exists on the
operator's machine. The shape-dependent guards are parameterised, but the claim is untested until
a real run.

---

## Open Questions

1. **Does `gate x mu` replace the mean?** views-hydranet#337 says it might, substantially. Owned by
   views-hydranet; this ADR changes only if that lands as a declared `aggregate_method`.
2. **Should the pipeline write this file itself** once the Track B memory behaviour is fixed,
   making §1 a stopgap? The converter would then become the reference implementation of stage 5
   for `stochastic` runs rather than a delivery tool.

---

## References

- `tools/collapse/` — the converters and the audit plots; `tools/collapse/__init__.py` has the
  usage and says which converter reads which source
- `tests/test_roster_conformance.py::test_collapse_declaration_matches_the_converter` — the pin
- `tests/test_collapse_darts_predictions.py::test_a_multi_sample_cell_becomes_the_mean_and_NOT_the_first_draw`
  — §6.3's guard against the `float(x[0])` hand-over
- views-models **#505** — the technical specification this implements
- views-models **#533** / epic **#532** — §6, the r2darts2 `dataframe` source
- views-models **#492** — the `prediction_frame` migration, which would move the darts models onto
  the §1–§5 path and make §6 a transitional section
- views-hydranet@32bc509 — `views_hydranet/utils/inference_orchestrator.py:175,177,179`
  (stages 4 and 5), `views_hydranet/utils/volume_handler.py:502` (`collapse_to_point`),
  `views_hydranet/utils/feature_scaler.py:199` (why invert precedes collapse)
- views-hydranet **#337** — `gate x mu` vs the draw mean
- Register: **C-47** (Track A/B dual output; why `skip_predictions_delivery` is `True`),
  **C-53** (the same flag silently regressed in a cross-branch merge)
