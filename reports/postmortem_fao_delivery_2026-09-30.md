# Post-mortem — the FAO delivery effort, 2026-09-29/30

> **DRAFT 1, written while the verification run is still in flight.** The rehearsal that tests
> the conclusions below started 2026-09-29T22:48:37Z and had completed one of eight models when
> this was written. **Nothing here is validated by a successful delivery yet**, and the sections
> marked *Pending* are the ones the run will decide. Expect at least two revisions.
>
> Sibling documents, deliberately not merged into this one:
> `reports/postmortem_runpod_first_deployment_2026-09.md` owns the **calibration** campaign —
> a different effort, different deliverable, different audience. This document owns the
> **FAO delivery leg**, which that one correctly flagged as untested (§: *"Nothing here has
> been tested on the forecasting partition"*).

---

## 1. What this effort was

On 2026-09-29 the first end-to-end FAO delivery was attempted and did not land. This effort is
everything between that failure and a chain that can be run again: three upstream defects found
and fixed across three repositories, a pinned build moved, four defects found in this repository
by falsification, and the delivery chain itself written down as a script for the first time.

**The single sentence that explains most of it:** *every fix that made a run work existed only
as commands typed into a rented pod, and the pod was destroyed.*

---

## 2. Timeline

| When | What |
|---|---|
| 2026-09-29 | First end-to-end FAO delivery attempted. Publish tore a partial commit; the delivery was not servable. |
| 2026-09-29 | `views-hydranet` sniffer defect found and fixed (#390), released **0.1.2**. |
| 2026-09-29 | `views-pipeline-core` content-hash dedup defect filed (#551), fixed (#552), released **3.3.4**. |
| 2026-09-29 | `views-postprocessing` findability defect filed, fixed (#314/#315), released **1.4.0**. |
| 2026-09-29 | `VIEWS_POSTPROCESSING_PIN` 1.1.1 → 1.4.0 in both launchers (#522). |
| 2026-09-30 | `/falsify` on "ready for a 40-lesson run" → **FALSIFIED**, three hard findings (#516, #517, #523). |
| 2026-09-30 | Guard-mode audit of the fixes' own tests → **10 of 14 guards decorative**. |
| 2026-09-30 | The FAO chain scripted for the first time (#524); `views-datafactory` floored (#509). |
| 2026-09-30 | Second `/falsify`, on the forecasting claim → preflight checked **3 of 9** publish variables (#527). |
| 2026-09-29T22:48Z | 40-lesson rehearsal launched on a fresh pod. **In flight.** |

---

## 3. The defects

### 3.1 Upstream, in three repositories

**`views-hydranet` — forecasting could not run at all.** The data pipeline passed
`is_forecast=True` to `sniff_forecast_alignment`, whose forecast branch requires a volume
beginning one month *after* the frame. The pipeline passes `VolumeHandler.from_df(df)` — the
history volume, which begins where the frame begins. **The condition was unsatisfiable**, so
every forecasting run raised "Forecast Continuity Broken" after a full training run. No
forecasting run had ever completed.

The guard and its test were added together and neither could observe the other failing: the test
mocked `DataSniffer` and asserted only that the flag was *passed*, never that the check it
selects can *pass*. This is the same defect class as §4.2 below, arriving from a different repo.

**`views-pipeline-core` — an upload reported success having written nothing.** Content-hash
dedup returned the pre-existing file's id. Worst case is **0 of 110 objects servable**, and the
documented remedy for an invisible delivery — re-publish — *was the trigger for it*. Registered
**views-models C-155** (Tier 1, pending on the unmerged #520 branch) and
**views-pipeline-core C-335** (Tier 1, registered 2026-09-29).

> *Cite the repository with the number.* This paragraph originally said "Registered C-155, Tier 1"
> and the `views-pipeline-core` seat read it as pointing at **their** C-155 — which is an unrelated
> **Tier 4 documentation-drift** row. Both entries are real and both describe this defect, in their
> own registers. Register IDs are per-repo on this platform and collide constantly; a bare `C-nnn`
> in a cross-repo document is ambiguous by construction. The adjacent **views-pipeline-core C-336**
> is the two-guards-masking-each-other finding that came out of the same fix.

**`views-postprocessing` — a findability guard that checked 2 of 110 artefacts.** 1.4.0 verifies
a delivery **by what it refuses**. A run that stops with `DeliveryNotFindableError` naming every
object is that build *working*; read as a failure it invites exactly the wrong remedy.

### 3.2 In this repository, found by falsification

**#516 — a fresh environment built successfully and WRONG.** Measured, not reasoned about:

```
declared:  views-datafactory>=1.9.0,<2.0.0 + numpy>=1.26.4,<2.0.0
resolved:  numpy 1.26.4, xarray 2025.12.0, pandas 3.0.6
the pod that delivered:  xarray 2024.3.0, pandas 1.5.3
```

The issue as originally filed called the environment *unbuildable*. It is not — it builds, on a
different pandas **major**, in silence, which is the more dangerous of the two shapes. A build
failure is loud. `views-datafactory` requires xarray outright and pandas only in an optional
extra that is not installed, so **xarray is the sole carrier** and pinning the carrier fixes it.

**The obvious fix would have been wrong**, and this is worth keeping: `xarray<2025` looks
correct, but **2024.11.0 already requires `pandas>=2.1`**. Measured boundary — 2024.3.0 is the
last release accepting pandas 1.x, 2024.5.0 moved to `>=2.0`, 2024.9.0 to `>=2.1`. **The cliff
is inside the 2024 line.** A year boundary is a plausible-looking guess; it is not a measurement.

**#517 — the pod could not publish.** The string `appwrite` appeared **nowhere** in
`pod_run_model.sh`. `views-pipeline-core[appwrite]` was never installed, so `_build_datastore`
raised at publish — after the full training run. That is precisely how 2026-09-29 failed, unfixed
in the repository, because it had been fixed by hand on the pod.

**#523 — a deliberate cheap run was reachable only by deleting the guard against an accidental
one.** The `>= 300` floor was right and stays. But it was written as though *wasted money* were
the risk, and it is not the main one: **the risk is an undertrained model reaching a partner as a
real forecast.** On that axis the guard made things worse, because the only way past it produced
output *indistinguishable* from a production run.

**#509 — the credential floor.** Before `views-datafactory` 1.13.0 the client could carry a netrc
credential across a redirect to another host and embed it in error messages. Every leg that runs
on rented hardware holds that credential. The pod installed `>=1.9.0`.

**#499 — the chain itself was never written down.** Nothing in `tools/` drove the FAO delivery.
`pod_run_model.sh` is hardcoded to `-r calibration -t -e` and its header is accurate that it
uploads nothing. The forecast-and-publish chain was typed by hand.

---

## 4. Where the process failed, which matters more than the defects

### 4.1 The audit examined the artefact in front of it

The first `/falsify` pass audited `pod_run_model.sh` — the script that *existed* — and concluded
we were three fixes from ready. It missed that the FAO delivery needs a **different chain
entirely**, because it never asked *what does this task require?*, only *is this script correct?*

**Auditing the artefact in front of you cannot find a missing one.** This is the most transferable
finding in the document.

### 4.2 The guards for the fixes protected nothing

An independent guard-mode audit reverted **every fix** in the commit, kept the comments, and got
**22/22 green**. 20 of 24 mutations survived; **10 of 14 guards were decorative**, 4 weak, none
held.

The mechanism is the lesson. Every assertion read the script as **text**, and `pod_run_model.sh`
is unusually well commented — each fix carries a paragraph naming the incident and quoting the
exact strings. **So the better the comment, the weaker the guard.** Deleting the code left the
comment, and the comment satisfied the assertion.

Worst survivor: `os.environ.get("REHEARSAL_LESSONS") or ""` → `or "40"`. **One word.** Every
production run then patches itself to 40 lessons, the floor is dead, and because the *shell*
variable stays empty the manifest says `mode: production` and no marker is written. A 40-lesson
model labelled a production delivery, suite fully green.

And one guard **certified a safety property that did not exist** — it claimed the RunPod guide
documented the `/workspace` chmod trap. It did not. The regex matched because `/workspace` is on
line 104 and `chmod` on line 180, joined by `.*` under `re.S`. That is worse than no guard,
because it stopped the next reader looking.

**The repository had already named this trap three times** — `pod_run_model.sh` refuses to
text-match a config citing #501 *"the guard that was not one"*, and insists the manifest report
the gated value rather than re-derive it because *"two readings of one number can disagree"*.
The author's own model named the trap and stepped into it. That is the argument for the
independence rule, not a slogan.

### 4.3 Preflight checked 3 of 9 publish variables

`PredictionStoreConfig` requires **nine**; only three are secrets. A missing *identifier* fails
the publish exactly as hard as a missing secret. The check would have passed with six of nine
absent and the run would have died at the publish — **the precise failure that mode exists to
prevent, introduced into that mode.**

Adding the six names would not have fixed it. The extra can be absent, the endpoint unreachable,
the key expired. Preflight now **constructs the config**, which answers the question instead of a
proxy for it. Suggested by the `views-pipeline-core` session.

**And the same check exists upstream, one layer down, with the same flaw** — filed as
`views-pipeline-core` **#557**. That module's own docstring reads:

> *"Reads Appwrite environment variables once at startup and fails loud if any are missing —
> preventing silent failures after hours of training. Addresses C-11."*

It is not called at startup. It is called from `_build_datastore()`, whose first caller is the
publish step — after `_train_ensemble` and after every constituent's two-hour subprocess. **The
module is the failure it claims to prevent, in that concern's own words.** On 2026-09-29 the
missing `appwrite` extra surfaced there: after the full training run, at publish, on rented GPU.

Two things make this worth more than its severity. The guard is **correct** — it does fail loud,
and `PredictionStoreConfig.from_environment()` refuses a short environment exactly as designed. It
is invoked at the wrong *time*, which no test of the guard can detect. And the docstring asserting
the defect is handled is the part that does the damage, because it **stops the next reader
looking** — the same failure as the pre-fix dedup contract and the `limit(1)` comment in §3.1, and
the same failure as the `re.S` regex in §4.2. Three instances in one document of *a correct
statement that has silently stopped being true*.

### 4.4 Verification order, twice

A guard was proven by breaking the code and confirming the test failed — but the revert
(`git checkout --`) ran while the *fix* was still uncommitted, discarding it. **Twice.** The
`ship-it` procedure already specifies mutation-verification *after* the commit, for exactly this
reason. **The ordering is the control; care is not a substitute for it.**

### 4.6 Nobody owned the seam

*(Numbered 4.6, not 4.5: the `views-postprocessing` seat's section — a guard that fires, passes
its test, and names the wrong cause — was written into PR #528 as 4.5 at the same time this was
written onto the branch as 4.5. Two seats independently reached for the next free number in a
document neither of them owned, which is a small instance of the thing this very section is
about. Renumbered here rather than there because #528 is the one still in review.)*

Hours before the run, `views-pipeline-core` ran a six-probe falsification of *"we are ready for a
full 40-lesson run"* and returned **SURVIVED**. This session ran its own pass and returned
**FALSIFIED**. Both were correct.

Every probe in the first pass held, and two of them were load-bearing: the sidecar is uploaded by
`views_pipeline_core.modules.appwrite`, so the fix released hours earlier was the code that would
run; and the pinned `views-postprocessing 1.1.1` requires `views-pipeline-core >=3.0.0,<4.0.0`, so
that pin blocked *its own* repo's fix and not the other's. Neither of those is guessable from
outside that repo.

The three actual blockers were a lesson-count guard, an unpinned resolve landing on pandas 3, and
`appwrite` never installed on the pod — all in `views-models`, none visible from the other seat.
They shared one cause: **every fix that made the previous run work was applied by hand to a pod
that was then destroyed.** The pod was the artefact and nothing in git described it.

**This is not §4.1 restated.** §4.1 is a *scoping* failure — an audit examined the artefact in
front of it. This is an *ownership* gap: four sessions each audited their own repository
competently, and the joins between them were nobody's. A scoping failure is fixed by widening the
question. An ownership gap is not fixed by anyone being more careful, because no participant's
diligence covers ground they cannot see.

Two consequences worth separating:

- **A verdict must carry its own scope in the same breath.** "SURVIVED" answered a question about
  one repository inside a sentence about the run. The limits *were* stated in the same message and
  it did not matter — a one-word verdict travels and its caveats do not. Where no one owns the
  join, saying so **is** the finding, not a caveat beneath it.
- **The joins need an owner before they need a process.** This arrived three times in one day from
  three directions — the `opponent` session on the pre-registration, the sidecar probe above, and
  the three blockers. Three independent arrivals is not a coincidence; it is the shape of the
  platform.

---

## 5. What went right

- **Four sessions across four repositories**, each fixing its own repo, none editing another's.
  `views-pipeline-core` declined to build a guard off a peer request because it was an ADR-013
  contract change and a maintainer's call. That was the correct refusal.
- **The best probe of the night came from a peer**: `views-pipeline-core` checked *which
  component actually uploads the sidecar* rather than assuming. Had the answer gone the other
  way, 3.3.4 would have been irrelevant and the run would have failed identically.
- **Peer review corrected two of this session's claims** — a relayed finding presented as
  independent corroboration, and a `get_queryset()` contract claim that was wrong at the tag.
- **The `--preflight` design paid for itself before the run started**: it found that the pod
  image ships **no conda**, which the postprocessor launcher requires. That would have killed the
  delivery at the last step, after every GPU hour.

---

## 6. What we still do not know — *Pending the run*

- Whether the pool, publish and postprocessor steps work at all. **None has ever run on this
  roster.** `rusty_bucket`'s forecasting log is from 2026-07-20 and lists a different roster
  entirely (`temporary_crane`, `temporary_fox`, `Deployment Status: shadow`).
- Whether 80 GB is the right disk floor. The retained size of a forecast is **unmeasured**.
- Whether a 300-lesson run fits in memory. A rehearsal proves the **chain**, not the **capacity**,
  and memory is where this platform has failed before.
- Whether the FAO can use what lands.

**Measured so far:** one model, 40 lessons, **24 minutes** — of which only **5 are training**.
The other 19 are data fetch, diagnostics, posterior sampling and saving, and do **not** scale with
lesson count. Extrapolating: ~56 min/model at 300 lessons, ~7.5 h for the roster.

---

## 7. Known gaps, stated rather than implied

- **ADR-064 entity coverage does not run for these models.** Not skipped-with-a-log-line: the
  sniffer is **never constructed** on the PredictionFrame path. Deliberate (ADR-042) and already
  recorded in ADR-064's own "deliberately not covered" section — but **#499 B3 asserted the
  opposite** and has been corrected. If a HydraNet forecast an entity absent from the last
  observed month, nothing here would refuse it.
- **A rehearsal is marked, not refused.** Its forecasts reach the FAO shelf and are servable.
  The refusal belongs in `views-pipeline-core` as *declare-or-refuse* (#523) — an ADR-013
  contract change across four repos, and a maintainer decision.
- **`views-datafactory` is declared two ways**, deferred with a named trigger (#509).

---

## 8. The systemic argument

The difficulty of this work is a signal about the codebase, not only about whoever did it. The
errors share a shape:

- changed **one of 35** copies of the same fact
- asserted **text** because the subject is a bash script
- **lost a fix** because testing it required mutating the file the fix lives in

Three faces of one property: **nothing here declares what a run is.** Lesson count, run type,
environment and chain all live in imperative shell or in Python that must be *executed* to be
read.

| Smell | What it forces |
|---|---|
| `views-datafactory` declared in 35 files | any change is a 35-file change |
| `total_lessons` only in a file; `main.py` has no override | a cheap run **requires** editing a tracked file → hand-edits → the pod becomes the artefact |
| config is executable Python | a value can only be known by running it → text guards are wrong by construction |
| behaviour in bash with hardcoded run types | untestable → guards degrade to `grep` |
| two environment mechanisms (conda *and* uv) | a pod can pass one leg and fail the other, invisibly |

And the one that is uncomfortable because the same hand wrote both halves: **this repository
compensates with unusually rich comments and many guards, and the comments then defeat the
guards.**

### The same shape in `views-pipeline-core`, which makes it a pattern rather than an anecdote

The section above is written about this repository. **It is not local to it.** The clearest
instance upstream was found on 2026-09-29 while diagnosing a different defect, and it is the same
property with a different subject:

**Which ensemble tier an ensemble runs on is not in its config. It is hardcoded in generated
`main.py`** — `EnsembleManager` or `PredictionFrameEnsembleManager`, chosen by an import line in a
scaffolded file. `managers/ensemble/context.py` says so explicitly: the prediction format is
*"**not** read from the config here"*, and the two managers resolve it separately.

Three consequences, all of them the §8 shape:

- **No config-time validator can see it.** A constituent declaring `prediction_format:
  "prediction_frame"` inside a DataFrame-tier ensemble is a contradiction no sniffer can detect,
  because half the contradiction is not in any config. That is `views-pipeline-core` #530: the
  members ran, wrote frames, exited 0, and the ensemble reported *"No prediction files found"* —
  a message about saving, when nothing had failed to save.
- **The per-model half of it validates clean.** `CoreConfigSniffer` checks `prediction_format`,
  validates its value, and requires its companion key. The failing config **passes every one of
  those checks.** The repository validates it, admits it, and then fails cryptically at runtime.
- **A remedy was proposed publicly and cannot work.** Three issues carried a promise of a
  config-time refusal before anyone noticed the tier is not in config. Filed as
  `views-pipeline-core` #533, which now carries the correction as well as the defect.

And the scaffold still emits the legacy tier as the active default while the stated direction is
the other one, so **every new ensemble starts on the wrong side of this** — `views-pipeline-core`
#126, acceptance criterion unticked.

Two repositories, independently arrived at, same month: **a fact that governs a run lives in
executable code rather than a declaration, so nothing can check it and the failure surfaces as a
confusing message somewhere downstream.** One repository doing this is a local habit. Two is the
platform's default, and the argument below applies to both.

**Proposed smallest change with the largest effect:** make lesson count and run type
**parameters** to `main.py` rather than file contents. That deletes the entire hand-edit class —
no config patching, no leftover-patch detector, no `--rehearsal` needing to rewrite a checkout.
This is an ADR-shaped argument and is not yet written.

---

## 9. Actions

| # | Action | Owner | State |
|---|---|---|---|
| 1 | Parameterise lesson count and run type | — | proposed, ADR not written |
| 2 | Declare-or-refuse for incomplete runs (#523) | maintainer to sequence | open, ADR-013 change |
| 3 | Fold #526 into the RunPod guide, one pass, after this run | this repo | open |
| 4 | Floor the remaining 34 `views-datafactory` declarations (#509) | this repo | deferred, trigger named |
| 5 | conda→UV: remove the conda preflight when it inverts (#525) | migration | open |
| 6 | `get_latest_file_id` ordering (views-pipeline-core#555) | that repo | open |
| 7 | Credential check runs *after* training, not at startup (views-pipeline-core#557) | that repo | filed this effort; fix is to resolve the store config before the roster runs, behind the flag |
| 8 | Three Appwrite clients (views-appwrite#173) | þing | deferred with trigger |

---

*Register entries for this effort are deferred behind a named trigger: **views-models** PR #520
merging, which already consumes views-models C-154 and C-155. Numbers in this document are written
`<repo> C-nnn` throughout for the reason given in §3.1 — the one place they were not, they were
misread within hours.*
