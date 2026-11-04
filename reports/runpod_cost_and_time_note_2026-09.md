# Running VIEWS models on rented GPUs — what it took and what it cost

**To:** Håvard Hegre
**From:** Simon
**Date:** 28-09-2026
**Subject:** Measured cost and wall-clock of running eight HydraNet models on rented cloud GPUs

**Method:** Contemporaneous. Every figure in §2 and §4 was measured on the machines described.
The only projection is §5, and it is labelled as one.

> This exists because we have never run the pipeline anywhere except on our own hardware, and
> because I had to. It is a record of what that cost, not an argument that we should do it
> routinely — §6 lists what it does not establish.

---

## 1. Executive Summary

I lost access to our server on 24 September and rented GPUs from a commercial provider
(RunPod) to produce forecasts that were already owed.

Eight models, trained from scratch on the full global land surface, cost **about $24** and
completed **inside one working day**. The same work on our own hardware would have been free but
was unavailable; done sequentially on one rented machine it would have taken roughly two and a
half days, so most of the saving came from running five machines at once.

The estimate I wrote the day before the run said 64 GPU-hours. The measured figure is **32**.

This is enough to say the approach works and what it costs for one model family. It is **not**
enough to cost a full monthly production run — see §6.

---

## 2. What it cost, measured

| | |
|---|---|
| Models trained | 8 HydraNets, 300 lessons each |
| Coverage | global land surface, 64,818 grid cells (the producer's extent; the FAO delivery is curated to 64,742 at the delivery boundary) |
| Wall-clock per model | **202, 272, 253 minutes** (three complete; mean **4.0 h**) |
| GPU-hours, all eight | **~32** |
| Price paid | **$0.75/hour** per machine, including storage |
| **Cost per model** | **~$3** |
| **Cost, all eight** | **~$24** |
| Machines used in parallel | 5 |
| Elapsed time, all eight | **under one working day** |
| Cost of the one unusable machine | **under $1** (see §3) |

---

## 3. What we projected beforehand, and what happened

A cost note written on 27 September, before any model ran, estimated:

> *8 models × ~8 h = 64 GPU-hours = $22 (community) to $47 (secure)*

Measured outcome:

| | projected | measured |
|---|---|---|
| hours per model | ~8 | **4.0** |
| GPU-hours, all eight | 64 | **~32** |
| cost (secure tier, which we used) | $47 | **~$24** |

The time estimate was about twice too pessimistic. The cost landed at roughly half the
secure-tier projection.

**One thing went wrong and is worth stating.** The first machine I rented was the cheapest that
met the obvious specification, and it was **25× too slow** — a model would have taken eight hours
of evaluation instead of one. I found this by deliberately running one throwaway model first,
which cost under a dollar and about forty minutes. Without that check I would have discovered it
six hours into a real run, having spent most of the budget. The cause is documented; the
selection rule that prevents it is now written down.

---

## 4. What it bought

| | |
|---|---|
| Per model | 13 prediction files, one per forecast origin |
| Rows per file | **2,333,448** (36 months × 64,818 cells) |
| Verification | zero duplicate keys, all values finite and non-negative, row counts and geography checked against observed data |
| Size delivered to my laptop | **~18 MB per model** |
| Full posterior retained | yes — compressed 236×, so the complete uncertainty distribution came home too, not only the point estimates |

That last row matters more than its size suggests: the researchers' files can be regenerated in
a different **summary form** — a different statistic over the same draws — without paying for
another run. It does not cover a different time period or forecast horizon; those need a new
run.

Three models are complete and verified on my laptop; five were still running when this was
written.

---

## 5. What a full monthly run would cost — an extrapolation

**This is an extrapolation, not a measurement.** It assumes the remaining work behaves like the
work we measured, which is exactly what §6 says we have not shown.

Taking 4 GPU-hours per model at $0.75/hour, and assuming the other model families cost broadly
what HydraNet costs:

| | |
|---|---|
| The 8 HydraNets | ~$24 |
| All 117 models in the platform, if they behave similarly | **~$350** |
| Elapsed time with 5–8 machines in parallel | **1–2 days** |

Assumptions this rests on, stated plainly:

- **Only HydraNet has been measured.** The stepshifter, r2darts2 and baseline families are
  assumed to be similar and have not been run on rented hardware at all. Many are far cheaper;
  none has been timed there.
- **The ensemble step has never been run on rented hardware**, and **the delivery step has
  never been run on rented hardware.** More importantly, neither is priced by model count:
  they run once per production run, and the pooling step is **memory-bound rather than
  GPU-bound** — our own configuration records it peaking at ~28.6 GB. So no part of the figure
  above speaks to them. They are a different rental line item, not a small increment on it.
- A production forecast run is a **different shape** from what we measured — one forecast origin
  rather than thirteen — and is likely cheaper per model, but this is untested.
- Machine availability fluctuates minute to minute. A monthly run needs a fallback rule for
  which machines are acceptable. That rule now exists in writing.

---

## 6. What this does not show

- It does **not** show that a full monthly production run can be done this way. It shows that one
  model family can be trained and its predictions retrieved.
- It does **not** cover publishing to our data store. Nothing was published from rented hardware,
  deliberately: a machine we do not control should not hold credentials that write to systems our
  partners read.
- It does **not** account for my time. The $24 is machine rental. The first day included a
  wrong machine, a diagnosis, and building the tooling — none of which recurs, but none of which
  was free either.
- It is **not** a comparison with our own server, which remains cheaper per run and is the right
  home for routine work. This was a response to losing access to it.

---

## 7. Honest uncertainty

- Five of the eight models were still running when this was written. The cost figures are based
  on three completed models; if the remaining five differ materially I will reissue this note.
- The per-model figure comes from three observations spanning 202–272 minutes. That spread is
  real and I do not yet know what drives it.
- This note draws on a working document
  (`reports/postmortem_runpod_first_deployment_2026-09.md`) that is still marked draft. The cost
  and timing sections of it are settled; other sections will change.

---

*Technical detail, including what went wrong and what we learned, is in
`reports/postmortem_runpod_first_deployment_2026-09.md`. The procedure for doing this again is in
`docs/runpod_run_guide.md`.*
