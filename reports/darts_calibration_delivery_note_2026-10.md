# Darts deep-learning models — calibration forecasts, global PRIO-GRID month

**To:** Håvard Hegre  
**From:** Simon  
**Date:** 10-10-2026  
**Delivery:** `darts-pgm-calibration-2026-10` — please quote this if you refer back to it; the
September delivery had no such label and we have since been unable to establish what was in it

---

## 1. What these models are

All nine are **off-the-shelf deep-learning models for time series**, taken from the Python library
**Darts** (`unit8co/darts`). The architectures are the library's, unmodified — each is a published
model used as Darts implements it. What we supply is the data, the configuration, and for six of the
nine a loss function of our own (§4).

They are *global* models in the forecasting sense — one model is fitted across all grid cells at
once, rather than one model per cell.

| | |
|---|---|
| **Data backend** | the dataFactory, for all nine |
| **Spatial coverage** | global land — all 64,818 PRIO-GRID land cells |
| **Temporal coverage** | the **calibration** partition: trained on Jan 1990 – Dec 2017, forecasting **Jan 2018 – Dec 2021** (`month_id` 457–504) |
| **Outcomes** | state-based, non-state and one-sided fatalities (`lr_ged_sb`, `lr_ged_ns`, `lr_ged_os`) |

---

## 2. The models

| model | type | loss function | notes |
|---|---|---|---|
| `dark_river` | N-BEATS | MSE | outcomes only |
| `old_rules` | N-BEATS | Spotlight | outcomes only |
| `blue_ocean` | N-BEATS | Spotlight | + 69 covariates |
| `silent_fox` | TSMixer | MSE | outcomes only |
| `dancing_monkey` | TSMixer | Spotlight | outcomes only |
| `brave_heart` | TSMixer | Spotlight | + 69 covariates |
| `red_hawk` | TiDE | MSE | outcomes only |
| `dark_necessities` | TiDE | Spotlight | outcomes only |
| `golden_eagle` | TiDE | Spotlight | + 69 covariates |

The names are arbitrary labels and carry no meaning.

**Within each block of three, exactly two settings vary** — the loss function, and whether the
model sees covariates. Everything else is identical: layer widths, depth, training epochs, learning
rate, early stopping. So inside a block, each row differs from the one above it in a single setting.
Across blocks the architecture changes and the comparison is a different one.

- **outcomes only** — the model's inputs are the past values of the three outcomes and nothing
  else. Purely autoregressive. Six models: the first two rows of each block.
- **+ 69 covariates** — the same three histories plus 69 predictors: ACLED event and fatality
  counts, land cover shares, population and built-up area, infant and child mortality, V-Dem
  governance indices, SHDI development indices, travel time to cities, mineral and petroleum
  deposits, growing season and harvested area, and terrain. Three models: the third row of each
  block.

Note that the two splits are not the same split. Six models are outcomes-only and six use the
Spotlight loss, but they are different sixes — `blue_ocean`, `brave_heart` and `golden_eagle` have
covariates *and* Spotlight, while `dark_river`, `silent_fox` and `red_hawk` have neither.

**There is no model combining MSE with the covariates.** Three configurations exist, not four.

---

## 3. The model types

All three are feed-forward neural networks built from ordinary dense layers. None uses recurrence,
convolution or attention.

### N-BEATS

A deep stack of fully connected layers with backward and forward residual links. It builds the
forecast as a sum of learned basis functions, which is what makes it partly interpretable. Designed
for univariate point forecasting and benchmarked on the M3, M4 and TOURISM competition datasets.

- Darts: [`NBEATSModel`](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.nbeats.html)
- Paper cited there: Oreshkin, Carpov, Chapados & Bengio (2020), *N-BEATS: Neural basis expansion
  analysis for interpretable time series forecasting*, ICLR 2020. arXiv:1905.10437

### TSMixer

Dense layers only, applied alternately along the time axis and across the predictor variables, so
the model mixes temporal structure and cross-variable structure in turn. Built for the multivariate
case.

- Darts: [`TSMixerModel`](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tsmixer_model.html)
- Paper cited there: Chen, Li, Yoder, Arik & Pfister (2023), *TSMixer: An All-MLP Architecture for
  Time Series Forecasting*, Transactions on Machine Learning Research. arXiv:2303.06053

### TiDE

An encoder–decoder, again dense layers only. It compresses the history and the predictors into a
single code, then decodes the whole 36-month path from it in one pass.

- Darts: [`TiDEModel`](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tide_model.html)
- Paper cited there: Das, Kong, Leach, Mathur, Sen & Yu (2023), *Long-term Forecasting with TiDE:
  Time-series Dense Encoder*. arXiv:2304.08424

### One setting all nine share

All nine use **reversible instance normalisation**: each cell's series is standardised before the
network sees it and the transformation is inverted on the way out. The model therefore fits on a
per-cell normalised scale, while the forecasts you have are in counts.

- Paper cited by Darts for all three models: Kim, Kim, Tae, Park, Choi & Choo (2022), *Reversible
  Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift*, ICLR 2022

---

## 4. The loss functions

**MSE** — ordinary squared error, every cell-month weighted equally. The library's own
implementation, a thin wrapper around PyTorch's. Used by `dark_river`, `silent_fox` and
`red_hawk`.

**Spotlight** (`SpotlightLossLogcosh`) — ours, not off the shelf. A three-part objective computed
after an `asinh` transform. It scores the *shape* of the error only in cells where the observation
or the forecast exceeds one fatality; separately penalises the *level* in cells whose observation
exceeds one fatality; and separately penalises forecasting conflict in cells that stayed below it.
It uses `log cosh` rather than squared error, so a single very large observation does not dominate
the gradient. Used by the other six.

### What you should know about Spotlight

It is Dylan's own construction, developed since March 2026; the variant used here dates from April
and is the fifth iteration of a family of related objectives. He has written a detailed account of
how it works. One of its two parameters is derived rather than tuned — the threshold separating
"event" from "quiet" cells is `asinh(1) ≈ 0.88`, exactly one fatality on the transformed scale. The
other, a weight on a spectral term, was chosen by hyperparameter sweep.

**What does not exist is any verification of it.** No citation anywhere — I searched the source,
its documentation, the commit history and the decision records; two other objectives in the same
in-house library do carry references to published papers, so this is not a codebase that never
cites anything. And no tests: the library's test suite covers eight of its objectives and this is
not among them. Nothing checks its gradients, compares it against a reference implementation, or
establishes that the code computes what the written account says.

That verification belongs with the author first — tests and a derivation someone else can follow
are what make a bespoke objective reviewable at all. Until they exist I am not in a position to
vouch for it. **It could be excellent, adequate, or quietly wrong, and nothing available to me
distinguishes those three.** That bears on how much weight a comparison between the Spotlight
models and the three MSE ones can carry.

---

## 5. The files

**One folder per model.** Inside each: thirteen Parquet files, `…_00` through `…_12`, and a short
text file recording when the model ran, which version of the software produced it, and which
configuration it used.

**The thirteen files are thirteen forecast origins**, stepping forward one month at a time. An
origin is the last month the model was allowed to see; from there it forecasts the next 36 months.

| file | last month seen | months forecast |
|---|---|---|
| `…_00` | Dec 2017 | Jan 2018 – Dec 2020 |
| `…_01` | Jan 2018 | Feb 2018 – Jan 2021 |
| … | … | … |
| `…_12` | Dec 2018 | Jan 2019 – Dec 2021 |

Within every file the earliest month is horizon 1 and the latest horizon 36, so a given target month
sits at a different horizon in each file containing it. Metrics computed per origin keep those
apart; metrics pooled across origins mix them.

**Each file holds 2,333,448 rows** — 36 months × 64,818 cells — and five columns:

| column | contents |
|---|---|
| `month_id` | month |
| `priogrid_id` | grid cell |
| `pred_lr_ged_sb` | forecast state-based fatalities |
| `pred_lr_ged_ns` | forecast non-state fatalities |
| `pred_lr_ged_os` | forecast one-sided fatalities |

**Units are fatality counts, untransformed** — not logged, not scaled. A value of 2.4 means 2.4
fatalities. Values are continuous, not integers.

Nine models × thirteen origins = **117 files, about 2 GB** in total.

### Observed values are not in this folder, and do not need to be

The evaluation joins forecasts to observed fatalities on `month_id` and `priogrid_id`. Those
observations are not included here, and Sonja's ensemble-construction tool does not need them to
be: it fetches actuals itself from a viewser queryset named in its own configuration, and also
accepts an actuals dataframe passed in directly. They can equally be pulled from the dataFactory.

Three things about that tool are worth flagging to whoever runs it, from reading its current source
rather than from memory:

- **The file naming already matches.** It expects
  `predictions_calibration_{timestamp}_{ts}.parquet` with `ts` running `00`–`12`, which is exactly
  what is here.
- **Its shipped default configuration is country-month, not PRIO-GRID.** The default keys are
  `("country_id", "month_id")` and the default actuals queryset is a country-level one. The
  evaluation code itself is indifferent to the level — it merges on whatever keys the configuration
  names — so pointing it at these models is a configuration change, to `("priogrid_id", "month_id")`
  and a PRIO-GRID actuals queryset, not a code change.
- **It resolves prediction files inside the repository layout**, at
  `views-models/models/<model>/data/generated/`, rather than from an arbitrary folder. The files
  here are laid out one folder per model and will need placing accordingly, or the paths adjusting.

If you would rather hand it to a research assistant than to the tool, the instruction is: *iterate
over the thirteen files per model, join each to the observed values on `month_id` and
`priogrid_id`, and compute point metrics per origin so horizon effects stay separable.* Parquet
reads in one call — `pandas.read_parquet` in Python, `arrow::read_parquet` in R.

### What is not included

| | |
|---|---|
| observed values | fetched by the tool, or from the dataFactory (above) |
| evaluation metrics | none computed — see below |
| training logs | not shipped. `calibration_log.txt`, which the ensemble tool can optionally read to resolve run timestamps, is among them; timestamps can be passed explicitly instead |
| other partitions | calibration only — no validation partition, no true forecasts |
| model artefacts | no weights, no checkpoints |

---

## 6. Additional notes

**No model here has a predictive distribution.** These are point forecasts — one number per
cell-month, no draws, no quantiles, no intervals. All nine are configured as deterministic
predictors, so a different export would not recover a distribution; it was never produced. Nothing
that scores a *single model's* uncertainty — CRPS of its predictive density, interval coverage,
density calibration — can be computed from these files.

One qualification, since it would otherwise look like a contradiction in the output: the ensemble
tool can still produce a CRPS. It does so by treating the member models' point forecasts as an
empirical sample and scoring the spread across models. That is a property of the ensemble, not of
any model in it, and it is not the same quantity as a CRPS computed from a model's own posterior.
Worth knowing before the two are compared.

**Nothing here has been evaluated.** I checked only that the files are not broken: no negatives, no
missing values, no duplicate cell-months, the expected row counts, and totals within the same order
of magnitude as observed. That rules out a failed back-transformation or a collapsed model and
establishes nothing else.

**It is nine models, not eleven.** Two others were set up to draw 100 samples per forecast, making
them probabilistic counterparts of `dancing_monkey` and `dark_necessities`. A point delivery needs
one value per cell, and at one sample their configurations are byte-identical to those siblings — the
same model twice. Running them as intended is blocked by memory: 100 samples for 64,818 cells across
thirteen origins needs far more RAM than any machine we have.

**One observation will dominate any squared-error metric.** December 2021, cell 148759 has **113,395**
observed fatalities; the next largest anywhere in the window is 5,395. That month falls in only one
of the thirteen origins (`…_12`), where that single cell is about 41% of observed fatalities. I do
not know whether it is a real event or a data error. Either way it is better settled before the
numbers are produced than explained afterwards.

**Column naming matches the tool.** These files use `pred_lr_ged_sb`, which is the ensemble
tool's own default `target_column`. The September delivery used `pred_lr_sb_best` instead, so if
anything needs renaming when the two are compared it is that one, not this. Renaming either takes
minutes and needs no re-run.

**Comparison with the September delivery.** You were sent three variants per model then — a mean, a
95th percentile and a conditional mean — each computed from that model family's predictive
distribution. These nine have no distribution, so only the mean has an equivalent here. If the
evaluation work has been built on one of the other two, the two deliveries are not like-for-like.
A correction while I am here: my September note said the full distributions had been retained so
alternative summaries could be produced without re-running. They were not kept.
