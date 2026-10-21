# Per-head gate calibration — `sb`, `ns`, `os`

**Date:** 2026-10-21 · **Cost:** $0 (local CPU over retrieved weights) · **Seat:** views-models
**Status:** measurement only. This seat does not author views-hydranet preregs or outcomes.

## Why this exists

Every gate-calibration number in the #565 programme — σ̂, every `a` and `b`, and the **−71%**
temperature result — comes from one line, `2026-10-12_underfire_diagnosis/p2_gate_calibration.py:119`:

```python
gate[d, fed - BASE_ORIGIN] = torch.sigmoid(out.cls[0, SB])[lri, lci].numpy()
#                                                     ^^^^
```

`SB` is the fixed index `0` (`vehicle.py:53`). The model declares **three** classification targets
(`config_hyperparameters.py:210`):

```python
'classification_targets': ['by_sb_best', 'by_ns_best', 'by_os_best'],
```

So `ns` and `os` had never been measured, and a temperature fitted on `sb` had no claim on them.
The question that prompted this: *"is the recommendation that we implement temperature scaling on
all HydraNet such that sb, ns and os are all correctly calibrated?"*

## Method

`sweep_all_heads.py` is a copy of that harness's forward sweep with the head index lifted to a
loop, so all three heads are collected in **one** pass (the forward pass is the cost; three heads
instead of one is free). Truth per head is built exactly as `p2_gate_calibration.py:124-128` does
for `sb`, using `TARGETS[idx]`. The data is the existing `pull_land_all3_121_504.parquet` cache —
it already carries all three features, so no viewser pull was needed.

⛔ **Parity gate.** Before anything is written, the `sb` cube this harness produces must equal the
committed `p2_result_<arm>.npz` cube under `np.array_equal`. **It does, for every arm swept** — gate
and truth both. Without that, no `ns`/`os` number here would be comparable to the published `sb`
numbers.

The remedies are then fitted by `tier2_remedies.py` **unmodified except for the directory it reads
from** (`EV`), so `R1` (temperature), `R3` (Platt: slope + intercept) and `R4` (isotonic) and every
falsifier are the committed statistics. Fit on 12 origins, evaluate on the held-out 1, 13 folds.

views-hydranet was read-only throughout: `git status --porcelain` clean after every step.

## Base rates — the first reason one `T` cannot serve all three

| head | positives (13 origins × 64,818 cells) | base rate | vs `sb` |
|---|---|---|---|
| `sb` | 2,905 | 0.003448 | — |
| `ns` | 1,012 | 0.001201 | **2.9× rarer** |
| `os` | 1,088 | 0.001291 | **2.7× rarer** |

`T` is fitted against observed positive frequency. Three heads with base rates differing by ~3×
have no reason to share a slope correction.

## Results

Three distinct seeds. `amber`, `silver` and `silver014` are **bit-identical** (190/190 weight
tensors), so they contribute **one** seed, not three; `bronze` is 43 and `cobalt` is 44. All are
`weight_decay = 0.1` controls. `b = 1.000` is an honest gate.

| arm | seed | head | positives | T | T spread | raw b | b after T | F2 | raw err | err after T | err after Platt | Platt b |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| amber | 42 | **sb** | 2,905 | 0.8708 | 0.8% | 1.1917 | 1.0378 | pass | 0.00981 | 0.00287 (-70.7%) | 0.00335 (-65.9%) | **1.0037** |
| amber | 42 | **ns** | 1,012 | 0.9249 | 1.6% | 0.9832 | 0.9096 | ⛔ **FIRES** | 0.00446 | 0.00326 (-26.9%) | 0.00300 (-32.7%) | **1.0176** |
| amber | 42 | **os** | 1,088 | 0.9117 | 1.4% | 1.0186 | 0.9288 | ⛔ **FIRES** | 0.00814 | 0.00290 (-64.4%) | 0.00251 (-69.2%) | **1.0117** |
| bronze | 43 | **sb** | 2,905 | 0.8841 | 0.8% | 0.9915 | 0.8767 | ⛔ **FIRES** | 0.01286 | 0.00642 (-50.1%) | 0.00530 (-58.8%) | **1.0038** |
| bronze | 43 | **ns** | 1,012 | 0.9217 | 1.3% | 1.0268 | 0.9466 | ⛔ **FIRES** | 0.00404 | 0.00361 (-10.6%) | 0.00386 (-4.5%) | **1.0137** |
| bronze | 43 | **os** | 1,088 | 0.8724 | 1.1% | 1.0568 | 0.9221 | ⛔ **FIRES** | 0.00670 | 0.00249 (-62.8%) | 0.00406 (-39.4%) | **1.0056** |
| cobalt | 44 | **sb** | 2,905 | 0.8479 | 1.1% | 1.0173 | 0.8627 | ⛔ **FIRES** | 0.02045 | 0.00569 (-72.2%) | 0.00298 (-85.4%) | **1.0038** |
| cobalt | 44 | **ns** | 1,012 | 0.7905 | 1.6% | 0.8895 | 0.7034 | ⛔ **FIRES** | 0.02884 | 0.00665 (-76.9%) | 0.00327 (-88.7%) | **1.0169** |
| cobalt | 44 | **os** | 1,088 | 0.8885 | 1.2% | 1.0100 | 0.8976 | ⛔ **FIRES** | 0.00779 | 0.00232 (-70.2%) | 0.00286 (-63.3%) | **1.0092** |

F2 fires in **8 of 9** cells.
Platt b range: **1.0037 .. 1.0176** (max |b-1| = 0.0176)
Temperature b range: 0.7034 .. 1.0378 (max |b-1| = 0.2966)
All dAP: [0.0]
raw b across seeds, sb: 1.1917, 0.9915, 1.0173   range 0.2002
raw b across seeds, ns: 0.9832, 1.0268, 0.8895   range 0.1373
raw b across seeds, os: 1.0186, 1.0568, 1.0100   range 0.0468

ACROSS-SEED spread of the fitted T (F9's 15% criterion is WITHIN-arm and does not cover this):
  sb: 0.8708, 0.8841, 0.8479   spread 4.2%
  ns: 0.9249, 0.9217, 0.7905   spread 15.3%  <- EXCEEDS F9's 15%, across seeds
  os: 0.9117, 0.8724, 0.8885   spread 4.4%

Within-arm fold spread (what F9 actually tests), max across all cells: 1.6%  -> F9 passes everywhere

## What it says

**1. `ns` and `os` have no slope defect.** Raw `b` for them spans 0.8895–1.0568. The quantity
temperature corrects is not broken in those heads, so correcting it moves them away from calibrated.

**2. The `b = 1.1875` premise is a seed-42 property.** `sb` raw `b` is **1.1917** at seed 42,
**0.9915** at seed 43 and **1.0173** at seed 44 — a range of **0.2002**, which is larger than the
0.1875 defect the whole track is named after. Independently consistent with σ̂_b = 0.0374–0.1107
(views-models #572): seed 42 is the distant draw.

**3. `F2` fires in 8 of 9 cells.** It passes only on `amber`/`sb` — the single cell the published
**−71%** was measured on. F2 is this programme's own criterion for *"temperature is the matching
fix"* (|b after T − 1| < 0.05).

**4. Temperature is not stable across seeds either, for `ns`.** The `T spread` column is the
**within-arm 13-fold** spread, which is what `F9` tests, and it passes everywhere (max 1.6%).
**Across seeds** `T` for `ns` spans 0.7905–0.9249, a spread of **15.3%** — which would exceed
F9's own 15% criterion if F9 were applied in that direction. It is not; F9 never looks across seeds.

**5. Platt holds everywhere.** `b` after a two-parameter fit lands in **1.0037 … 1.0176** across
all nine cells — max |b−1| = **0.0176**, against **0.2966** for temperature. Temperature has only a
slope knob (`sig(x/T)` cannot move the intercept); `ns` and `os` need an intercept, which is the
mechanism, not a curve-fitting preference.

**6. ⚠️ ΔAP = 0.00000 in all nine cells**, against a seed sd of 0.019, with rank identity `True`
under both `R1` and `R3`. **No map moves.** `F7` already fires on `sb` for this and the prereg's
own word is *cosmetic*. Any per-head calibration work is reporting integrity, not forecast skill.

## Scope and limits

- Three seeds, one architecture, `land`, the 13 out-of-sample h=1 months 456–469.
- `copper` (pos_weight 2.0) and `indigo` (weight_decay 0.0) were **not** swept per head — they move
  a second parameter. Free to add on request.
- No estimator was selected by this seat: `b` is `tier2_remedies.py`'s own `ml_fit` throughout.
- ⛔ **This seat makes no claim about what the prereg should say.** F2 firing is reported as a
  measurement against the programme's own stated criterion. Whether that retires the temperature
  remedy, re-scopes it to `sb`, or triggers an amendment belongs to the views-hydranet seat, which
  is why this is raised there as an issue and not written into their documents.

## Artifacts

`results/p2_result_{amber,bronze,cobalt}_{sb,ns,os}.npz` (gitignored — regenerable, see below) ·
`heads_iso/sweep_all_heads.py` · `heads_iso/perhead_table.py` · `heads_iso/HARNESS_SOURCES.md` ·
`perhead_table.txt` · `sweep_cobalt_allheads.log`

```bash
cd heads_iso
python sweep_all_heads.py <artifact .pt> <arm> ../results/p2_result_<arm>.npz   # ~9 min, parity-gated
for h in sb ns os; do python tier2_remedies_perhead.py <arm>_$h; done
python perhead_table.py
```
