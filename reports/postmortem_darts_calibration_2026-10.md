# Post-mortem — the pgm darts calibration campaign: nine of eleven, and three defects that only running could find

**Status:** Final
**Dates:** 2026-10-05 → 2026-10-09
**Epic:** [#532](https://github.com/views-platform/views-models/issues/532) · tracking [#539](https://github.com/views-platform/views-models/issues/539)
**Delivered:** 9 distinct models × 13 rolling origins × 2,333,448 rows, global pgm — retrieved off the pods and verified on the operator's laptop
**Spent:** **~$17** of rented GPU time, as two allocations (~$12.70 of $15 on 10-07/08, then ~$4–5 of a $10 top-up on 10-09). Not reconciled against RunPod's billing — the pod-hours are known, the invoice is not. Broken down in §1
**Related:** `postmortem_runpod_first_deployment_2026-09.md` (the first rented-hardware campaign, whose rules this one inherited and in one case got wrong), `postmortem_fao_delivery_2026-09-30.md`

---

## Why this exists

The eleven pgm `views-r2darts2` models had been on `development` since 2026-09-19 and **not one had ever produced a prediction**. Two attempts had failed: the laptop's 31 GB ran out twice (#489), and a run on `fimbulthul` completed while writing nothing, because the engine labelled the prediction index `country_id` and every refusal was swallowed by a thread pool (#504, views-pipeline-core#529).

Researchers — Håvard directly, Sonja through the model-selection work — were waiting on calibration-partition point predictions.

This records what it took, which was more than running them.

---

## 0. Timeline

| when | what |
|---|---|
| **10-05 22:30** | Epic #532 filed: 6 stories, tracking issue, after an investigation that found the models already correctly configured and nothing able to run them |
| **10-06** | #533 converter, #534 pod runner, #535 docs written. Separately: #531 diagnosed — **not** the env_path bug it was reported as — fixed in PR #540 to `staging_202608`. #541 and views-pipeline-core#558 filed |
| **10-07 19:23** | #536 merged: the two 100-sample models aligned to the point delivery |
| **10-07 19:35** | #543 merged: the chain. First pod rented |
| **10-07 ~23:15** | `dark_river` trains and predicts 13 origins in 87 min — **the first pgm r2darts2 prediction in existence** — then **fails in the conversion step** that turns the engine's output into the delivery parquets |
| **10-08 09:27** | #544 merged: the scale guard made frame-level |
| **10-08 ~08:00–12:00** | Four pods, eight models queued. Five complete. **Two turn out to be byte-identical to siblings.** `blue_ocean` sits 100 min at 0% GPU. Budget exhausted at ~$12.70; three runs killed 20–40 min from completion |
| **10-09 01:44** | #545 merged: scratch moved off the network mount, heartbeat added |
| **10-09 ~05:00** | `blue_ocean` reaches the GPU in **8 minutes** and finishes in 97. `golden_eagle` follows |
| **10-09 ~07:25** | `brave_heart` dies: `NaN in SpotlightLossLogcosh` |
| **10-09 10:13** | #547 merged: `brave_heart`'s missing `target_scaler` |
| **10-09 ~11:45** | `brave_heart` completes in 104 min. **Nine of eleven** |

---

## 1. The headline

**Nine distinct models — every architecture in every training configuration that exists.**

| | MSELoss, outcomes only | Spotlight, outcomes only | Spotlight, + 69 covariates |
|---|---|---|---|
| **NBEATS** | `dark_river` | `old_rules` | `blue_ocean` |
| **TSMixer** | `silent_fox` | `dancing_monkey` | `brave_heart` |
| **TiDE** | `red_hawk` | `dark_necessities` | `golden_eagle` |

Three architectures × three training configurations. **Not a complete cross** — there is no model combining MSELoss with the large predictor set, so three configurations rather than four.

Two corrections to how I described this roster during the campaign, both config-review findings (§3.3, #546) rather than model findings. I called it *3 architectures × 2 feature sets* for four days: the predictor count is the difference you notice reading the configs, and `loss_function` is what actually separates these models. And I called the two predictor sets *3 and 71 variables*: the real figures are **3** — the histories of the three outcomes and nothing else, so six of the nine are purely autoregressive — and **72**, those three plus **69 covariates**.

Runtimes clustered tightly: **87–114 minutes** across the nine, with the three covariate-rich models (97, 98, 104 min) indistinguishable from the rest once their real blocker was removed. (The two duplicates ran 107 and 116; they are excluded here because they are not deliverables.)

**What the hardware was, and what it cost.**

| | |
|---|---|
| Machines | RunPod, GPU `NVIDIA RTX PRO 4500 Blackwell Server Edition` |
| Selection rule | RAM ≥ 50 GB · vCPU ≥ 12 · VRAM ≥ 24 GB, ranked RAM first — inherited from the September campaign, where a cheap 31 GB pod ran 25× slower through kernel reclaim without ever OOM-killing |
| Pods rented | **6** across three days; up to **4 in parallel** |
| Bare pod → first training step | **5 min 32 s** (measured on the first pod: system packages 78 s, Python environment 147 s, environment verification 69 s, config checks 20 s) |
| Model wall-clock | 87, 87, 97, 98, 99, 99, 101, 104, 114 min for the nine — mean 98 (the two duplicates ran 107 and 116) |
| Model-minutes delivered | **886** for the nine — about **15 GPU-hours** |
| Spent | **~$17** total, ≈ **$1.25 of pod time per model delivered** |
| Wasted | **three runs killed 20–40 minutes from completion** when the first budget ran out, having already consumed most of their cost and returned nothing |

The **5 min 32 s** is the number worth keeping: it is what the #534 runner bought. A pod becomes useful almost immediately, which is what makes short-lived rented machines viable — though in practice we queued several models per pod rather than renting one each, because the setup cost is small but the rental clock is not.

**Three defects stopped this campaign, and every one was found by a run failing rather than by a check. Two of the three were mine, introduced during this campaign.** That is the finding worth keeping; the models are the deliverable.

---

## 2. What was built

| story | artefact |
|---|---|
| #533 | `tools/collapse/collapse_darts_predictions.py` — list-in-cell → one scalar per cell |
| #534 | `tools/podrun/pod_run_darts_calibration.sh` — bare pod → verified delivery parquets |
| #535 | `docs/runpod_run_guide.md` Phase 4c, `docs/ADRs/024_pod_runner_contract.md` |
| #536 | the two 100-sample models aligned — **and this is the change that created two duplicates** |

**#543 shipped three stories in one PR**, against the epic's own one-PR-per-story rule. Recorded in the commit rather than done quietly: #534's tests import `EXPECTED_ROWS` from #533's module, so #534 alone is red. They were not independently mergeable.

### What we did not build, and are still standing on

Two upstream defects made the September run write zero predictions while reporting success. This campaign fixed neither. One we worked around; the other had been fixed upstream, in code, in the week before we ran — which I got wrong in an earlier draft of this document and corrected only on 2026-10-09.

| | state | what views-models does about it |
|---|---|---|
| [views-r2darts2#55](https://github.com/views-platform/views-r2darts2/issues/55) — `entity_id` defaults to `"country_id"` regardless of level, so a pgm model's predictions are labelled `country_id` and refused downstream | **open** | all eleven pgm models declare `"entity_id": "priogrid_id"` in `config_meta.py`, with `tests/test_darts_entity_id_matches_level.py` so a new pgm darts model cannot omit it. The reader already derives it correctly (`dataset/base.py:934`); only the writer does not |
| [views-pipeline-core#529](https://github.com/views-platform/views-pipeline-core/issues/529) — evaluation ran `validate_and_save` in threads and `concurrent.futures.wait()` discarded their exceptions, so a run that saved nothing reported PASS | **code fixed, issue still open** | nothing needed. The fix shipped before we ran |

#55 is a workaround that holds because a test holds it — and **that test checks the configs, not the engine**, so it cannot catch an engine-side regression. If #55 is ever fixed and someone removes the workaround, the engine-side test that would make that safe does not exist. There is also a twin one line above the defect: `time_id` is defaulted the same wrong way, harmless only because no darts model is yearly yet.

**#529 was closed in code between September's failure and this campaign, and the two sit on opposite sides of it.** `abe05c2` (29 September) replaced the `wait()` with `as_completed` plus `future.result()`, under a comment naming the September run: *"dark_river's thirteen sequences were all refused by CorePredictionSniffer, and the run reported PASS having written nothing... A guard whose refusal is discarded is worse than no guard."* First tag containing it is 3.3.1.

**Verified in our own environment, not inferred from a version range:** `dark_river`'s run log records `views-pipeline-core==3.3.4`, and the `as_completed`/`future.result()` form is present in that tag's `managers/model/model.py:1540`. The thirteen sequences then validated and saved. **So the invisible-failure mode was fixed before we started, and had any of our runs lost its predictions we would have been told.** The issue is open as bookkeeping lag, not unfinished work; someone should close it citing `abe05c2` and 3.3.1, which is pipeline-core's call.

That matters for what this campaign gets to claim. The runner's independent verification of the deliverable — count the parquets, check the rows, write `MANIFEST` only on that path — is still the right design and it is what caught the §3.1 collapse failure. But **it was not what stood between us and a silent loss of output, because the platform had already fixed that.** An earlier draft of this document said it was, which would have credited our diligence for someone else's fix.

What remains true: both defects live on the `prediction_format: "dataframe"` path, which is **80 of 117 models** on the platform (42 r2darts2, 38 stepshifter), and these nine were all on it. The `prediction_frame` path saves on the main thread and always did raise — a disk-full `OSError` failed a run loudly on 2026-09-20, which is the control case.

### The engine this campaign ran on, and who built it

Worth stating plainly because the post-mortem would otherwise take credit for it: **the engine work that made these runs possible was done the week before, in views-r2darts2, and almost none of it happened during this campaign.** Both 0.2.4 fixes were written on 29 September; the release commit is 30 September and the tag is 5 October. In the 1–9 October window r2darts2 contributed release mechanics — a version bump, a merge to `main`, the tag, two back-merges. The engineering was the week before; the availability was in ours.

What we were standing on, and neither fix is pgm-specific — pgm is only where one of them hurts:

- **`PredictionScratch` ([r2darts2#54](https://github.com/views-platform/views-r2darts2/issues/54)).** The memmap scratch directory had no owner and was never removed. It is **one directory per `predict()` call, not per run** — evaluation calls `predict()` once per rolling origin, so a 13-origin calibration leaves thirteen behind. Nothing removed them, and the accumulation filled a 2 TB disk on 2026-09-20. (Two figures circulate for that incident — ~4,000 directories, and ~53 GB per directory. They cannot both describe the same disk: their product is 212 TB. I have not reconciled them and am not going to assert either as the cause; what is established is that the directories were never freed and the disk filled.) The per-directory size scales with the grid, so at cm it is a nuisance and at 64,818 cells it is the disk. It deliberately does not free on garbage collection, because callers hold memmap views that outlive the owner; the `atexit` registration holds a strong reference on purpose.
- **A failed device restore now raises ([r2darts2#41](https://github.com/views-platform/views-r2darts2/issues/41)).** It used to log a warning and finish on CPU while the artefact still recorded a GPU. **This is the fix most specific to renting hardware:** a silent CPU fallback means paying GPU rates for CPU throughput and discovering it later as "why was this so slow". We never saw it fire, which is the point.
- **From 0.2.3 (19 September), without which nothing would have started:** the darts 0.40 / pandas 1 / zarr 2 pins that make it installable beside the platform at all; a `rich` fix without which every model died at epoch 1; an encoder-regeneration fix in streaming predict, without which evaluation failed for every model using cyclic encoders — 19 of the 31 cm darts models at the time; and toolz/dask import shims for Python ≥ 3.11.9. Dylan's entity-at-end filter, carried in from 0.2.2, is the genuinely pgm-shaped one — it stops a dense grid forecasting entities that no longer exist.

**Installing from the git tag was deliberate, not a workaround for a slow release.** 0.2.4 is intentionally absent from PyPI — confirmed, PyPI's latest is still 0.2.3 — because the device change can abort a pipeline that was previously limping along on CPU, and all 42 views-models configs pin `>=0.2.3,<0.3.0`. Publishing would have pushed that behaviour change to every one of them on their next install. The tag exists so that a machine can opt in without that happening, and #534 asserting the installed version is `0.2.4` is the other half of the same decision.

### The memory ceiling nobody owns

Work on #536 measured the `dataframe` path at **~23.3 GB per origin at 100 samples — ~303 GB for all thirteen held at once**, against a pod-selection rule of RAM ≥ 50 GB. That is why `mister_bluesky` and `little_talks` could not run as probabilistic models, which is the whole reason #536 existed and therefore the whole reason the two duplicates exist.

**That ceiling is not tracked anywhere.** It is in no views-r2darts2 issue and in no risk register; it appears only in the body of their PR #58 and a comment on their epic #57, both of which say the within-run peak was explicitly **out of scope** for the #54 leak fix — correctly, because freeing a directory earlier does not help when evaluation accumulates every sequence's frames before converting any of them. Capping it needs an override of pipeline-core's per-origin streaming hook.

Our figures appear to be the first real measurement of it, and they are corroborated from the other side: pipeline-core's own audit measured `to_prediction_df` peaking at +4,766 MB with 2,265 MB never returned to the OS at roughly half pgm scale, and its ADR-053 states that 64 samples at pgm scale exceeds workstation RAM. **So the leak across runs is fixed and the peak within a run is unowned.** C-155 is views-models' record of it; filing it upstream is a cross-repo decision and Simon's to make.

### The detour that was not in the plan

Two days in, Dylan and Håvard were blocked on the **cm** darts models on `staging_202608` — reported as a wrong `env_path`. It was not: `views-pipeline-core` became `optional=true` in views-r2darts2 0.2.0 and is supplied only through the `[manager]` extra, so eight models installed without it and could not import it. Fixed in #540.

Two things came out of that detour, which cost most of the morning of 10-06:

- **the diagnosis mattered more than the fix** — patching the reported `env_path` would have left it broken, or masked it;
- **it is not eight models, it is forty-three.** [#541](https://github.com/views-platform/views-models/issues/541) records 35 more on that branch with the same defect, and [views-pipeline-core#558](https://github.com/views-platform/views-pipeline-core/issues/558) records the cause: the scaffold template drops extras, so **every r2darts2 model it generates is born unable to import pipeline-core.** We fixed the eight people were standing on and filed the rest rather than quietly fixing forty-three in a delivery PR.

### Why the converter was not optional

`views_r2darts2`'s `dataframe` path writes 13 parquets that are *almost* the deliverable — except every cell holds a Python list, and the keys are an index. ADR-023 records what the consumer does with that:

> `_as_float_prediction_array` in `ensemble-updater` takes `float(x[0])` on a list cell — **one draw, silently**.

An unconverted hand-over does not fail. It publishes draw zero as the answer. For a deterministic model that is accidentally correct, **which is worse**, because the error surfaces only on the models where it does damage.

---

## 3. The three defects

### 3.1 The scale guard refused correct output after 87 GPU-minutes

`dark_river` trained, predicted all 13 origins, then died:

```
origin 06: largest collapsed 'pred_lr_ged_os' is 11.46, below 12
```

The output was correct. The threshold was inherited from the HydraNet converter, and **its own docstring said it had "never been validated against a darts run"**.

What the run measured:

| target | max | predicted/observed |
|---|---:|---:|
| `pred_lr_ged_sb` | 283.68 | 0.40 |
| `pred_lr_ged_ns` | 21.56 | 0.24 |
| `pred_lr_ged_os` | 14.94 | 0.26 ← refused |

`sb` at 283.68 settles the frame: were that still on the `asinh` scale the count it stood for would be `sinh(283.68)`, around **10^123 fatalities**. The output is therefore not still transformed. All three targets share one scaler chain. One-sided violence is the rarest of the three targets, and a small total for it is consistent with a point model on a rare target. **The diagnosis is that the scaler inverted correctly — not a judgement about the model.**

**The fix was the premise, not the number.** The per-target form is correct for HydraNet, which has a per-target scaler *registry* — one target can be stranded in log space while its siblings invert. r2darts2 has no such registry, so that failure cannot occur and the check only imported a false positive. Now frame-level (#544), with the surrendered guarantee asserted in a test so it cannot be lost quietly.

### 3.2 Scratch on the network mount — the expensive one, and mine

`blue_ocean` ran 100 minutes at **0% GPU and 11.6% CPU**, never reached the GPU, and was killed having produced nothing.

The cause was #534, four days earlier. The disk floor measured `$ROOT` while the engine wrote into `TMPDIR`; a floor that measures a filesystem the workload does not use cannot fire, so #534 set `TMPDIR=$ROOT/tmp` to make the check meaningful.

**The principle was right and the application was backwards.** It moved the *workload* to the filesystem the *check* already watched. `$ROOT` is `/workspace`, a **network filesystem** — the property that makes `chmod` silently fail there (C-154), documented on the page #534 was editing. With the outcomes-only predictor set that is ~1 GB, and five models completed without anyone noticing; with the covariate set it is ~15 GB pushed over the network.

**Three hypotheses were measured and discarded before this one:**

| suspect | measured |
|---|---|
| `filter_frame_entities_at_end` boolean mask | 4.5 s, and drops 0 rows on a dense grid |
| `_first_value` calling `dropna()` once per column, 148 times | 0.3 min |
| `_frame_to_grid` reindex | 11.2 s |

~30 seconds, not 100 minutes. What reframed it was the **CPU number**: 11.6% with 0% GPU is a process blocked on I/O, not computing.

**Confirmed by the fix.** With scratch on local disk, `blue_ocean` reached the GPU in **8 minutes** and finished in 97.

```
before   gpu=0%   cpu=11%   ← blocked on the network
after    gpu=14%  cpu=91%   ← working
```

### 3.3 `brave_heart` had no `target_scaler`

Died after ~90 GPU-minutes with `NaN in SpotlightLossLogcosh: per_channel=[nan, nan, nan]`.

It was **the only one of 42 darts models declaring a loss function and no target scaler**. Three siblings run the *same* loss and trained fine. The engine scales the target through that key alone (`dataset/base.py:1192-1223`), so the loss saw raw counts reaching **113,395**, and `logcosh` evaluates `cosh(x)`, which overflows float32 above ~89:

```
log(cosh(asinh(113395))) = 11.64   finite
log(cosh(113395))        = inf  →  NaN
```

MSELoss would merely have been badly conditioned. **The loss decides whether the omission is survivable**, which is why `silent_fox` (MSE) and `dancing_monkey` (same loss, scaler present) both survived.

**What hid it:** `brave_heart` carries a `feature_scaler_map` explicitly listing `lr_ged_sb/ns/os` under `AsinhTransform`. It reads exactly like target scaling. It is not — that map applies to columns used as *features*. A convincing near-miss, and presumably how it passed review.

---

## 4. Where the process failed

More important than the defects, per this repository's convention. Seven failures below; five of them are mine (§4.1, §4.2, §4.3, §4.5, §4.7). §4.4 was a design decision of mine that cut both ways, and §4.6 was a scripting error.

### 4.1 I made a guard meaningful by moving the work instead of moving the guard

§3.2. The reasoning — *a floor must measure what the workload writes* — was sound, and I had just registered C-156 about exactly that class. I applied it in the direction that required no further thought, and never asked whether the destination was a suitable place to write 15 GB. **The guide said it was a network filesystem on the page I was editing.**

### 4.2 A test that was true and measured the wrong property

#536 set `num_samples: 100 → 1` and `mc_dropout: True → False` on two models. I checked what *depended* on those keys — no ensemble pinned them, no test referenced them. I never asked whether they were **the only thing distinguishing those models from a sibling**. They were.

The guard I wrote asserted the delivery batch was internally *comparable*. **Identical is maximally comparable**, so it passed. A test can be true, pass honestly, and be measuring something other than what you care about.

Found only because the run produced byte-identical output on two different pods (`corr = 1.0000`). Registered as C-158.

### 4.3 I planned a campaign on one measurement

`dark_river` took 87 minutes. I projected the campaign from that single number. The second round ran ~150 minutes — and I killed three models **20–40 minutes from completion** because the budget was gone. At roughly $1.25 of pod time per model, the three killed runs had already consumed most of that and returned nothing.

One measurement is a data point, not a distribution. The nine delivered runs spanned 87–114 minutes, so the per-model error was not large — the error was *planning as though it were zero*.

### 4.4 The self-shutdown protected the budget and stranded the data

I built the pod to power itself off when its queue finished, so it could not idle at $0.75/hr while the operator slept. It worked. It also powered off with `old_rules`' output still on it, because I asked *"can this waste money?"* and never *"what happens to the output of whatever is running when it shuts down?"*

Recoverable only because `poweroff` leaves the volume intact. **If I had put `terminate` in that script instead, the model would be gone.** Subsequent runs gate the shutdown on a download-confirmation file.

### 4.5 My mutation verification was unsound for same-length edits

While mutation-testing #536's guard I changed `"num_samples": 1,` to `"num_samples": 5,`, restored the source, **verified byte-identity by sha256** — and the next run still read 5.

`exec_module` reuses `__pycache__` when the cached bytecode's recorded source **size** and mtime still match, and those two strings are the same length. The source was correct and the check was reading something else. **13 test files and 3 production paths load configs that way**; the worst instance is `pod_run_model.sh`'s `--rehearsal`, which patches a config and re-imports it *to verify the patch took*, and is safe today only because `300 → 40` changes the length. Registered as C-156.

### 4.6 A branch delete ran after a failed merge

A network error killed the merge API call for #543; the `gh api -X DELETE` on the next line ran anyway, deleting the head branch and auto-closing the PR **unmerged**. Recovered — the commit was in a worktree — but the delete should have been gated on the merge result, and now is.

### 4.7 I put a model-quality claim in a deployment post-mortem

The first draft of §5 reported one model's sanity numbers as *the* magnitude result for the campaign and wrote that the models "under-predict". Two things were wrong with that, and the smaller one is that the numbers came from `dark_river` alone — the first model to finish — and were generalised to the other eight without re-measuring. Re-measured, the roster does not share a single direction, so the sentence was false as well as out of scope.

**The larger one is that no such claim belongs in this document at all.** This is a post-mortem about deploying eleven models on rented hardware: what we built, what broke, what it cost, what we would do differently. It is not an evaluation report. The platform has an evaluation framework — a primary ranking metric plus guardrails, scored on the **validation** partition against a baseline — and a ratio of summed counts on the **calibration** partition is one guardrail on the wrong partition. It can say "the output is conflict-shaped rather than garbage". It cannot say a model under-predicts, and it certainly cannot rank one model family against another.

The draft also compared these models' magnitudes to the HydraNets'. **That is removed and should not come back.** Comparing two model families on performance is the evaluation framework's job, in its own reports, on its own partition. What these two post-mortems may compare is the *deployment*: what the pods cost, which config fields caused confusion, which guards were inherited and whether they transferred. Those sections stay (§3.1 and §3.2 are exactly that). Model quality does not.

The per-model table moved to the delivery note, where the recipient needs to know what is in the files he is being handed, and where it is labelled as a magnitude sanity check rather than a result.

---

## 5. Did we check the output was not garbage?

Yes, once, cheaply, and that is all we did. **No model in this campaign has been evaluated.**

**Why we checked at all.** A parquet can pass every schema check and still be worthless. These files
have the right columns, the right dtypes, 2,333,448 rows and no duplicate keys — and all of that
would still be true if the inverse target transform had failed and we were shipping log-space
numbers, or if the model had collapsed to a constant. §3.1 is exactly that anxiety: a scale guard
fired on correct output, I changed the guard, and I then wanted independent evidence that I had not
changed away the thing it was there to catch. §3.3 added to it — `brave_heart` ran with a missing
`target_scaler` until the last morning, so "does the output look like fatalities" was a live question
for at least one model.

**What the check was.** Summed predicted counts against summed observed counts for the same cells
and months, per target, per model; plus every value finite, every value non-negative, and the
fraction of cell-months with a nonzero prediction against the observed fraction.

**Why that check and not something better.** It needs only the predictions and the cached actuals,
it runs on a laptop in three minutes, and it fails loudly in the one way we were worried about: a
stranded log transform shows up as totals orders of magnitude wrong, and a collapsed model shows up
as a nonzero fraction of 0% or 100%. It is one admissibility guardrail out of the platform's set,
computed on the **calibration** partition, which is not the partition governance ranks on. It was
chosen because it is cheap and because it catches the specific failure in front of us.

**What it said.** Nothing is garbage. Every value is finite and non-negative. Totals land within a
factor of a few of observed in both directions. Sparsity is in the right neighbourhood — nonzero
predictions in roughly half a percent of cell-months against roughly four tenths of a percent
observed. No model is constant, none is in log space, none is empty. The files are plausible enough
to be worth someone's time, which is the bar this check was set at.

**What it did not say, and cannot.** Whether any of these models is any good. Whether one is better
than another. Whether conflict is predicted in the right places. Those need the evaluation
framework's own metrics on the validation partition against a baseline, and none of that has been
run. **Nothing in this campaign supports a conclusion about model quality, and this section must not
be read as one.**

**One deployment finding did come out of it.** The numbers varied far more between models than I
expected, and the variation tracked a single hyperparameter — not the architecture and not the
predictor count, which are the differences you notice reading the configs. I had been describing
this roster as 3 architectures × 2 feature sets for the whole campaign, and that was simply wrong
about what distinguishes these models. That is a configuration-review problem, and it is the best
evidence #546 has: these config fields have large consequences and nothing reviews them as though
they do. **It is not a result about the models** — establishing what that hyperparameter actually
does requires the evaluation framework, properly, on the right partition, which is work nobody has
done.

### One observation dominates the extremes

Observed month 504, cell 148759 = **113,395** fatalities. The next largest in the window is **5,395** — a 21× jump. Month 504 falls in only one of the thirteen origins, `…_12`, and that single cell-month is ~41% of the observed fatalities in it. Either a catastrophic event or a data artefact. **It should be looked at before anyone quotes any max-value comparison**, and it is the same value that overflowed `brave_heart`'s loss.

### Two models were never independent

`mister_bluesky` ≡ `dancing_monkey`, `little_talks` ≡ `dark_necessities`. See §4.2. They were the probabilistic variants of their siblings, and removing the probabilistic part removed the model. The eleven were never eleven designs: they are 3 architectures × 3 training configurations, plus the 2 probabilistic variants that collapsed onto siblings when configured for a point delivery.

---

## 6. What went right

**The heartbeat paid for itself the day it was written.** After `blue_ocean`'s silent stall, #545 added a two-minute log line: GPU %, CPU %, RSS, scratch size. Those three numbers separate the states that look identical from outside — training, CPU-bound conversion, blocked I/O. The first stall took hours of code-reading and three wrong hypotheses. The next question was answered by reading one line. **It measures rather than asserts, which is why it found what five new guards did not.**

**Measuring before concluding.** Three hypotheses about the stall were discarded by measurement, not argument. The correct one came from a number I had been looking at and not reading (11.6% CPU).

**Inverting tests rather than deleting them.** Four tests became wrong during this campaign — the scale guard's per-target form, the `TMPDIR` direction, the "threshold is unvalidated" message, and the batch-comparability claim. Each was rewritten to assert the *new* intent and carry the incident that changed it, so the reversal cannot be silently re-reversed.

**Test-first on #547**, as asked: the guard was written before the fix, was red for exactly **one model out of 42**, and green after one line.

**The ritual caught things.** `review-diff` on #542 found that both models' sweep configs already declared the values the PR was setting — the configs had contradicted each other, which made the change a correction rather than a reduction. On #543 it found nothing; on #545 it found the MANIFEST's `scratch_peak` was never a peak.

**`#531` was diagnosed rather than accepted.** Reported as a wrong `env_path`; it was a missing `[manager]` extra, and fixing the reported symptom alone would have left it broken or made it worse.

---

## 7. Why rented hardware, what it was good at, and what we would do differently

**Why we rented at all.** Not preference. `fimbulthul` no longer resolves, so the server that ran the September attempt is simply not reachable; the laptop has 31 GB and ran out twice on these models (#489). Renting was the only remaining way to produce predictions that were already owed. This is the second campaign in three weeks to reach that conclusion, which is itself worth noticing — two deliveries in a row have now depended on a commercial provider because our own hardware was unavailable.

**What renting was genuinely good at.**

- **Parallelism on demand.** Four pods ran at once. On one machine these nine models are about fifteen GPU-hours serially; we got them in two working days around a budget interruption and a flight.
- **Start-up is nearly free.** 5 min 32 s from a bare pod to the first training step (§1). That is what makes per-model disposable machines viable, and it is the single best argument for this pattern.
- **A throwaway first model is cheap insurance.** `dark_river` alone, on one pod, produced the first honest runtime and the first real peak-RAM and peak-disk numbers. It also found the scale-guard defect (§3.1) before eight other models were in flight.
- **The blast radius of a mistake is one machine.** Three of this campaign's four process failures (§4.1, §4.2, §4.4) would have been worse on shared hardware.

**What it was bad at.**

- **The filesystem is not the one you think it is.** `/workspace` is a network mount. It silently ignores `chmod` (C-154) and it is slow enough that putting scratch on it stalled a model for 100 minutes at 0% GPU (§3.2). Neither is discoverable without measuring.
- **The clock is always running.** A budget ending mid-run destroys the run and its cost: three models were killed 20–40 minutes from completion, having consumed most of their money and returned nothing. On our own hardware they would have finished.
- **Teardown is adversarial to data.** Protecting the budget by self-powering-off stranded a completed model's output on a volume (§4.4). The two concerns pull in opposite directions and the runner now resolves them by gating shutdown on confirmed retrieval.
- **Nothing hand-done survives.** A fix applied by hand on a pod dies with the pod, so every fix had to become a commit before it was worth anything — which was good discipline and also pure friction.
- **It makes a silent CPU fallback expensive.** On our own machine, finishing on CPU is slow; on a rented GPU it is slow *and* billed at GPU rates. That is why r2darts2#41 matters more here than anywhere else (§2).

**What we would do differently.**

1. **Measure two models, not one, before planning the campaign.** `dark_river`'s 87 minutes became the plan; the second round ran ~150 and the budget did not survive it (§4.3). One number from one model on one pod is not a rate.
2. **Ask for the budget the measurement implies, before starting.** `dark_river` established ~$1.25 of pod time per model, which puts eleven models at ~$14 of successful runs before any waste. That figure was knowable the moment the first model finished and was never put to Simon as a single number. Instead the money ran out mid-run twice, and the campaign in fact cost ~$17 for nine models — the difference being the three runs killed near completion and the stalled `blue_ocean` attempt.
3. **Do not run four pods against one datafactory and assume independence.** The second round's ~150 minutes against the first's ~100 is still unexplained and contention is the obvious candidate (§8). Either stagger them or measure it.
4. **Write the runner's disk and memory floors against the filesystem the work uses, from the start.** §3.2 is this lesson and it cost a model.
5. **Verify the deliverable outside the framework from the first run, not after the first loss.** This one we got right, and only because September had already taught it.
6. **Keep a cost ledger as the campaign runs.** Writing this on 2026-10-10 I can give pod-hours but not a reconciled invoice, which is why the most precise honest statement about this campaign's cost is "roughly $17".

**Would we do it again?** For an owed delivery with no server, yes, and the tooling is now in place to make it cheap to start. As routine practice, no — our own hardware is cheaper per run and does not bill for mistakes. The September cost memo said the same thing and this campaign did not change it.

---

## 8. What we still do not know

- **Whether these predictions are any good.** Entirely open. That is what the delivery is for, and it is not this campaign's question.
- **What cell 148759 at month 504 actually is** — a catastrophic event or a data error. 113,395 fatalities in one cell-month, 21× the next largest in the window.
- **Whether the network-filesystem diagnosis is complete.** The fix worked and the before/after is unambiguous, but we never A/B'd it on identical inputs; three other hypotheses were measured and this is what remained.
- **Whether `brave_heart`'s numbers are sound.** It trained and produced valid output, but its config was wrong until the morning of 2026-10-09 and nobody has looked at whether its predictions are sensible beyond the schema checks.
- **Why the second round ran ~150 min against the first's ~100.** Datafactory contention across four pods is the obvious candidate and was never measured.
- **Whether the three covariate-rich models gain anything.** Unknown, and this campaign cannot say — we measured their runtime and their disk footprint, not their worth.
- **What `loss_function` actually does to these models.** The sanity check (§5) showed it dominates the differences between them, which told us our description of the roster was wrong. What it means for model behaviour is an evaluation question and is untouched: it needs the framework's metrics on the validation partition against a baseline. Until then we know the field matters and not what it does. Feeds #546.

---

## 9. Rules to adopt

1. **A resource floor follows the work; the work does not move to the floor.** §3.2 cost a model to learn. Generalises past disk: the in-code memory guard reads host RAM and approves runs that cannot fit a container.
2. **Before normalising a field across models, ask what else that field distinguishes.** §4.2. "Make these consistent" can mean "delete one of them".
3. **A long-running job logs a heartbeat of what it is waiting on.** Silence is the one failure a log cannot explain.
4. **Gate teardown on confirmed retrieval, never on completion.** §4.4.
5. **Commit before mutating, and never mutate a config by a same-length edit without clearing `__pycache__`.** §4.5.
6. **Gate a destructive follow-up on the result of the action it follows**, not on reaching the next line. §4.6.
7. **One measurement is not a distribution.** Plan the second run; do not plan the campaign. §4.3.
8. **A claim about N models is measured on N models.** If the table has one row and the sentence says "they", the sentence is wrong. §4.7.
9. **When two changes touch one file, run the merge result, not just each branch.** Not ours, but earned next door and transferable: r2darts2's #54 and #41 each passed alone and failed together — the leak tests fed a CPU-reporting mock while the device guard raised on a GPU host, and CI being CPU-only kept it green until a developer machine with CUDA found it. #543 shipped three interdependent stories in one PR for a related reason (§2).
10. **A deployment post-mortem does not evaluate models, and never compares model families.** §4.7. It may ask "is this output conflict-shaped or is it garbage", because that is a question about the pipeline. Ranking, skill and cross-family comparison belong to the evaluation framework, on the validation partition, in its own reports. Two post-mortems may compare deployments — cost, config confusion, which guards transferred — and nothing else.

---

## 10. What closed, and what opened

**Closed:** #533, #534, #535, #536. And **epic #488's definition-of-done line 4** — *"at least one of the 11 passes calibration end-to-end on a GPU"* — which had been open since 2026-09-19 and this campaign satisfied.

**Opened:**

| | |
|---|---|
| [#541](https://github.com/views-platform/views-models/issues/541) | 35 more models on `staging_202608` carry the same missing-`[manager]` defect |
| [#546](https://github.com/views-platform/views-models/issues/546) | **Model config fields have no schema** — four instances in four days: a missing key, a look-alike key, a dead key, and a set that must agree |
| [views-pipeline-core#558](https://github.com/views-platform/views-pipeline-core/issues/558) | the scaffold template drops extras, so every r2darts2 model it generates is born unable to import pipeline-core |
| **C-154** | `/workspace` ignores `chmod` — written because the guide had cited it for ten days with no entry behind it |
| **C-155** | the `dataframe` tier's RAM ceiling: ~303 GB at 100 samples |
| **C-156** | stale `__pycache__` defeats every config check that loads by spec |
| **C-157** | config fields have no schema |
| **C-158** | two models distinguished only by sample-count keys |

**Open upstream, and what each means for these nine** — checked with the views-r2darts2 seat rather than recalled:

| | bearing on this delivery |
|---|---|
| [r2darts2#55](https://github.com/views-platform/views-r2darts2/issues/55) `entity_id` defaults to `country_id` | **load-bearing.** Our configs' explicit `entity_id` is the only reason the nine ran. Remove it and the runs break again, silently |
| [views-pipeline-core#529](https://github.com/views-platform/views-pipeline-core/issues/529) exceptions discarded in threads | **not live — fixed in code before we ran** (`abe05c2`, first tag 3.3.1; our pods installed 3.3.4). The issue is open as bookkeeping; pipeline-core should close it |
| [r2darts2#42](https://github.com/views-platform/views-r2darts2/issues/42) inverse-transform fall-through | **not exposed.** Latent and gated on a darts upgrade; only `MinMaxScaler` configs are at risk, and all nine delivered models use `AsinhTransform`. Across the 42 darts models, 40 declare a target scaler: **33 `AsinhTransform`** (one of them added by #547) and **7 `MinMaxScaler`**. Those 7 are the ones to check before anyone runs them |
| r2darts2 [#22](https://github.com/views-platform/views-r2darts2/issues/22), [#25](https://github.com/views-platform/views-r2darts2/issues/25) cache-path and cache-filename re-derivation | no effect observed; both bypass pipeline-core's staleness check, so a stale cache would be used silently |
| [r2darts2#56](https://github.com/views-platform/views-r2darts2/issues/56) | ADR-017 should say the entity guarantee is ingest-time, not predict-time |

**Should be opened and is not:** the within-run memory peak (§2). Cross-repo, so Simon's call — the r2darts2 seat has offered to file it with our numbers attributed.

**Where to find what a darts run is contractually supposed to produce:** views-r2darts2's documentation pass of 29 September — one frame per target, shape `(N, S)` with `S` read off the array, count space clipped at zero so a zero may be a clipped negative, which entities are present, and that the on-disk format is pipeline-core's decision. We had been inferring most of that from code.

**Still open and blocking the last two models:** [#492](https://github.com/views-platform/views-models/issues/492). Until the `prediction_frame` migration lands, `mister_bluesky` and `little_talks` cannot run at 100 samples, and at 1 sample they are duplicates. **The delivery is nine, and saying "eleven files" would be false.**

**#546 is the one that matters.** Three of this campaign's defects were config fields that no schema describes, and each was found by a run failing. The argument in that issue is that adding one more bespoke guard per incident scales linearly with keys and finds nothing in advance — a key that *is* read is exercised by every run, while a key that is *not* read is exercised by nothing.
