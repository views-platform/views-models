# ADR-015: Posterior Sample-Count Standard and the Ensemble Constituent Contract

**Status:** Accepted
**Date:** 2026-06-26
**Deciders:** Simon, VIEWS platform team
**Related ADRs:** [ADR-013](013_regression_target_name_agnosticism.md) (config is the single source of truth — derive, never hardcode), [ADR-014](014_reconciliation_composition_root.md) (composition-root contracts); views-pipeline-core `prediction_frame_ensemble.py` (PFE concat), #143/#146 (FAO forecast ensemble)

---

## Context

A `PredictionFrameEnsembleManager` ensemble with `aggregation="concat"` pools its constituents' posterior draws by **concatenating the sample axis** (`np.concatenate([pf.values for pf in frames], axis=1)`, pipeline-core `prediction_frame_ensemble.py:99`). The pooled draw count is therefore the **sum** of the constituents' `n_posterior_samples` (empirically: `synthetic_chant`, 3 × 64 → 192). `rusty_bucket`, the FAO forecast ensemble (#143), pools 8 constituents × 128 → **1024** pooled draws, shipped uncollapsed to the summarizer.

Two problems motivated a standard:
1. **Sample counts were ad-hoc.** During integration, models declare low counts (16, 64) to run fast. Without a target, the pooled dimension is unpredictable and constituents weight the pooled mixture **unequally** — a constituent declaring 1000 draws dominates one declaring 10, even though both are equal members of the ensemble. (`golden_hour`'s 16/16/8 constituents are an example; its stale-artifact anomaly is tracked as #131/C-74.)
2. **The model/sample counts an ensemble expects were *inferred*, never *declared*.** Editing `config_modelset` silently changed the pooled shape with nothing affirming intent.

`np.concatenate` does not *require* equal counts — equality is a **fairness + predictability** standard, not a mechanical constraint.

## Decision

### 1. The integration-period sample-count standard is 128
For the current model-integration period, the go-to `n_posterior_samples` in views-models is **128**. It stays a per-model **hyperparameter** in `config_hyperparameters.py` — never a literal baked into logic (ADR-013 spirit). 128 is revisable; see Consequences.

### 2. Reconciling ensembles declare what they expect, explicitly
An ensemble may declare, in `config_hyperparameters.py`:
- `expected_models` — the number of constituents it expects, and
- `expected_samples_per_model` — the single sample count every constituent must declare.

These are **belt-and-suspenders**: `expected_models` duplicates `len(config_modelset["models"])` on purpose, so that touching the modelset forces a conscious re-affirmation of intent. The pooled total is then `expected_models × expected_samples_per_model` (e.g. 8 × 128 = 1024).

### 3. A config-time contract enforces the declarations (fail loud at CI)
`tests/test_ensemble_configs.py::test_declared_modelset_and_sample_counts_match_reality` asserts, for any ensemble that declares the fields (opt-in; legacy ensembles are skipped):
- `expected_models == len(config_modelset["models"])`, and
- every constituent's `n_posterior_samples == expected_samples_per_model` (derived via `conftest.get_n_posterior_samples`).

This pulls the check forward to CI — a mismatch fails in seconds, not at minute-45 of a monthly run.

### 4. A non-blocking report surfaces off-standard counts
`tests/test_sample_count_standard.py` emits a **warning** (never fails) listing sample-producing models whose `n_posterior_samples != 128`, so drift is visible without crying wolf. A true **runtime** log-warning on forecasting/production runs is a pipeline-core follow-up (views-models cannot add runtime behaviour without touching the immutable `run.sh` / per-model `main.py`).

### 5. The count reader is family-agnostic and divergence-guarded (amended 2026-07-20, register C-104)
Posterior sample count is named differently by each family's runtime — baseline `n_samples`, hydranet `n_posterior_samples`, r2darts `num_samples`, stepshifter `pred_samples` — while the runtime object and the ADR-013 wire already agree on one name (`PredictionFrame.sample_count`). `conftest.get_n_posterior_samples` therefore reads **whichever** sample-count key a config declares, rather than forcing a cross-repo rename (prefer-agnostic-over-uniform). A config that declares **multiple** of these keys with **different values fails loud**: that divergence is the decoy trap that silently discarded a sample-count change during the 2026-07-20 FAO delivery (a baseline config carries both `n_samples`, the runtime key, and `n_posterior_samples`, this contract's key, kept equal only by hand). The authoritative check remains the produced `pf.sample_count` at aggregation (pipeline-core, register C-85); this getter guards the config layer.

### 6. The count that matters is the count a model PRODUCES (amended 2026-08-10, Epic #242 S4)

A HydraNet family head draws `K` samples from the per-cell distribution on each of `D`
MC-dropout passes, so the **emitted** sample-axis width per cell is **D×K**, not `D`.
This is an observation about what the artifacts contain, verifiable here: the Epic #242
roster runs `D=4 × K=4 = 16`, and `tests/test_pfe_production_readiness.py` compares that
product against the on-disk `y_pred` width. It corroborates **views-hydranet ADR-067**,
which introduced the family head — that ADR is **`Proposed`, not Accepted**, in its own
repository, so this clause is stated on this repo's own evidence and does not depend on
its ratification. If ADR-067 changes, re-check the arithmetic here; nothing automatically
will.

**Consequence for §2 and §3.** "The sample count a model produces" — the number that must
equal `expected_samples_per_model` and the on-disk `y_pred` width — is the **produced**
count. `n_posterior_samples` alone (§5, `D`) is only one factor. The contract derives it
via `conftest.get_produced_sample_count` = `get_n_posterior_samples × get_head_sample_count`,
where `n_head_samples` **defaults to 1**, so every non-family model behaves exactly as it
did before this amendment. `rusty_bucket` declares `expected_samples_per_model: 16`, the
produced width.

**Consequence for §1.** The 128 standard is a target on the **produced** count, and the
roster is an order of magnitude under it (16 produced per constituent, 128 pooled across
eight). That gap is deliberate and has a cause outside this ADR: the full-depth run peaks
at ~28.6 GB and does not fit production hardware (thinned 2026-07-20). §4's non-blocking
report is what keeps it visible — it currently names 73 models. **This amendment redefines
the unit; it does not license the gap.** Register **C-91** tracks the tail-stability
consequence and was restated in the same change, because a standard whose unit moves while
the risk entry keeps the old numbers is how a measured concern quietly stops matching
anything.

**The two knobs stay distinct and each honest** — `n_posterior_samples` is the MC-dropout
depth, `n_head_samples` the family draws per pass, and the produced width is their product.
Neither is a rename of the other.

## Consequences

### Positive
- The pooled dimension is predictable and declared (`expected_models × expected_samples_per_model`).
- Constituents weight the pooled mixture equally (fairness).
- Editing a modelset without updating the declared expectations fails loud at CI.
- 128 stays a hyperparameter; nothing in logic branches on the literal.

### Negative / revisit
- **128 is on the low side for stable HDI tails.** For zero-inflated, heavy-tailed conflict posteriors, 95% credible-interval bounds from 128 draws are noisy. 128 is an *integration-period* standard; the production FAO delivery / the summarizer (views-frames#89) will likely want 512–1024 per constituent. Revisit before production.
- `expected_models` is intentionally redundant; the contract keeps the two in sync.

## References
- [ADR-013](013_regression_target_name_agnosticism.md) (derive from config, never hardcode), [ADR-014](014_reconciliation_composition_root.md).
- pipeline-core `managers/ensemble/prediction_frame_ensemble.py` (concat = `np.concatenate` on the sample axis).
- #143 (rusty_bucket), #146 (real constituents), #131/C-74 (golden_hour unequal-constituent anomaly).
