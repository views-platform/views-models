# The FAO delivery — Definition of Done, and the road to it

**Written 2026-09-29, at the end of the first end-to-end test.**
Author: this session. Audience: Simon, and whoever picks this up next.

> **The framing that matters:** none of what follows is data science. It is plumbing — packaging,
> credentials, contracts between repositories, and the machinery that moves a file from one place
> to another without lying about whether it arrived. You do not need to become a software engineer
> to own this. You need a definition of done you can check yourself, and a sequence someone can
> work through. That is what this document is.

---

## 1. Definition of Done

**Done is not "the code is fixed" and not "the test passed".** Done is: *FAO receives real
forecasts, and will keep receiving them.*

Concretely, all six of these are true at the same time:

| # | Statement | How YOU check it, without reading code |
|---|---|---|
| **D1** | FAO's API serves a forecast produced by a run we started that day | `GET /provenance/forecast` names today's run id |
| **D2** | It has the right shape | 3 targets (`lr_ged_sb/ns/os`), 36 months, 64,742 cells, `unmapped_count: 0` |
| **D3** | The numbers served are the numbers we produced | spot-check ~25 cells against what came off the machine — a script reports PASS/FAIL |
| **D4** | The whole chain runs from one command, with nothing hand-edited | one command, walk away, come back to a served forecast |
| **D5** | It is repeatable | run it twice from scratch; both complete; the second is identical to the first |
| **D6** | Someone who is not us can do it | a person follows the runbook on a fresh machine and succeeds without asking us |

**D3 and D5 are the two that matter most and are the easiest to skip.** D3 is the only thing that
proves the numbers were not altered in transit. D5 is the only thing that distinguishes "it worked"
from "it worked once".

**What is explicitly NOT in the definition of done:** forecast quality. Whether the models are any
good is a separate question, answered by evaluation, not by this. A delivery can be perfectly done
and carry poor forecasts.

---

## 2. Where we actually are

The 40-lesson test **succeeded at its job**, which was to find the breaks before the real run paid
for them. Six of the ten links worked for the first time in the platform's history:

```
 8 models forecast          WORKS  (first time ever)
 pooled to 128 draws        WORKS  (first time ever)
 published to the store     WORKS  (first time ever, and the #536 fix behaved exactly right)
 postprocessor delivers     WORKS  (first time ever)
 lands in FAO's bucket      WORKS  (first time ever)
 FAO ingests and serves     REFUSED
```

The refusal is correct behaviour. It declined rather than serving something it could not validate.
Had it accepted, we would have learned about it from FAO.

**Cost of learning all this: about $5 and one day.** The production run is $24. Finding these now
was the cheap outcome.

---

## 3. What is in the way

Grouped by what they actually block. **The severity ranking is by "would this break the real run",
not by how hard it is to fix.**

### Group A — blocks the production run. Must be fixed.

**A1. The sidecar can never be uploaded, on any run, forever.**
The delivery uploads a "sidecar" — a lookup table mapping grid cells to country boundaries. It is
*identical on every run*, because it describes geography, not forecasts. The storage system
deduplicates by content: seeing identical bytes, it declines to store a second copy and reports
success anyway. The manifest then points at a file that was never created, and FAO refuses the run.

This is not an accident of today. It will recur on **every** future delivery.

*The deeper problem:* the contract (ADR-013 §4.2) demands one sidecar **per run** for a file that
cannot vary by run. Forcing the upload would satisfy the contract by storing dozens of identical
copies. The contract is what is wrong.

**A2. An upload can report success having written nothing.**
The cause of A1 being invisible. The upload returned a real file id — for the wrong file — and the
caller took that as proof. Registered on this platform as C-94, and this is a new instance.

**A3. The delivery's own completeness check does not check the delivery.**
`_assert_delivery_is_findable` verifies two artefacts: the manifest and the historical file. It
does not verify the sidecar or the 108 shards — the things the manifest actually points at. So the
run reported "Completed" while being unservable.

### Group B — blocks a *repeatable* run. Must be fixed for D5.

**B1. The environment is not reproducible.** A dependency (`xarray`) floats unpinned. A fresh
install today gets a version incompatible with the platform's pinned pandas. Your laptop works only
because its environment is months old and has never been rebuilt. **Anyone building fresh, today,
fails.**

**B2. A pinned override silently reverts.** The `toolz` pin (register C-151) is applied after
install; any later install that re-resolves dependencies undoes it, with no warning.

**B3. The publishing machine needs a package nobody installs.** `pod_run_model.sh` omits the
`appwrite` extra, because it was written for a run type that never publishes.

**B4. Nobody has decided which machine publishes.** Your laptop lacks the memory, the server is
locked out, and rented hardware was excluded by a rule I wrote badly. **There is currently no
answer**, and every future delivery needs one.

### Group C — real, but not blocking. Track and schedule.

- **C1.** The Appwrite key carries 20 permissions and **expires 2026-11-17**. That date forces the
  work regardless; it is the moment to narrow rather than renew.
- **C2.** Three API endpoints gave three different answers about the same run. "Did it land?" is
  currently unanswerable from outside. *(faoapi's, being handled there.)*
- **C3.** No check anywhere asserts a forecast has all three targets. A one-target delivery would
  publish clean and report healthy. *(faoapi's, being handled there.)*
- **C4.** `/workspace` on rented machines silently ignores file permissions. Credentials placed
  there cannot be protected.
- **C5.** My own runbook's ground rule 5 is wrong and actively misleading.

---

## 4. The roadmap

Six phases. Each states what finishes it, so nobody has to guess.

### Phase 0 — Write it all down · **today** · me · *free*

File every finding as an issue in the repository that owns it; register the risk-class ones.
Nothing is fixed in this phase — the point is that none of it is carried only in this conversation.

**Finished when:** every item in §3 has an issue number or a register entry.

### Phase 1 — Finish the test · **optional, your call** · me · *~$1*

Use the stopgap faoapi identified (point the manifest at the sidecar that already exists) to get
the run ingested, purely to exercise **the last third of the chain, which has never run**:
do the served numbers match ours (D3), is the coverage right, does it look undertrained.

**Why it is worth doing:** there may be further breaks behind this one. Finding them now is far
cheaper than during the production run.

**Why you might decline:** it puts a run in a partner-visible store whose manifest references
another run's file. A deliberate provenance compromise, permanent in the listing.

**Finished when:** probes P4, P6, P8 and P10 have run and reported.

### Phase 2 — Fix the blockers · **the real work** · cross-repo · *free, but the longest phase*

In dependency order:

1. **A1 — the sidecar contract.** Decide whether the sidecar is per-run or shared. This is a
   cross-repo contract change (ADR-013, views-postprocessing, views-pipeline-core, views-faoapi)
   and is what the **þing protocol** exists for. *Everything else in this phase is small; this one
   is a decision that needs several repos to agree.*
2. **A2 — uploads must verify what they wrote**, by name, not by "did I get an id back".
3. **A3 — the completeness check must check every artefact the manifest names.**
4. **B1–B3 — pin the environment** so a fresh build is identical to a working one.
5. **B4 — decide the publishing machine.** Yours. See §5.
6. **C5 — correct the runbook.**

**Finished when:** each has a merged fix with a test that fails without it, and the packages are
released (pods install released versions, so an unreleased fix does not exist as far as a run is
concerned).

### Phase 3 — Prove it, cheaply · me · *~$4*

Re-run the full chain at 40 lessons, **twice, from a fresh machine**, with no hand-edits.

**Finished when:** D1, D2, D3, D4 and D5 all hold. This is the gate before spending real money.

### Phase 4 — The production run · me, on your go · *~$24*

Eight models at 300 lessons, pooled, published, delivered.

**Finished when:** D1–D4 hold on a 300-lesson run, and FAO is told it is real.

### Phase 5 — Make it routine · *ongoing*

The difference between "we did it" and "it keeps happening": the runbook updated so D6 holds, a
scheduled run, and an alert when a delivery goes stale. Plus the C-group items, with **C1's
2026-11-17 key expiry as a hard date**.

---

## 5. What is yours, and what is not

**Yours — nobody else can decide these:**

- **Which machine publishes to FAO** (B4). Not a technical detail: it is about what infrastructure
  this platform runs on. A personal laptop is not a defensible answer and you were right to say so.
- **The key**: narrow it, and by when — forced by the 2026-11-17 expiry.
- **Whether to take the Phase 1 stopgap.**
- **Priority between all of this and everything else you have.**
- Money, credentials, releases, and anything that reaches FAO.

**Not yours — do not let anyone hand these to you:**

- Which repository an issue belongs in, how a fix is designed, what a test asserts, how the
  contract should be restructured. That is engineering, and the sessions working these repos own it.

**The honest summary:** you are not out of your depth on the decisions above — those are judgement
calls about infrastructure and risk, and you have made several good ones today, including rejecting
the laptop and insisting on establishing behaviour before acting. You are out of your depth on the
mechanisms, and you do not need to be in it.

---

## 6. The one-line version

> Six of ten links work for the first time ever. One contract is wrong in a way that would break
> every future run, and three guards failed to notice. Fix those, pin the environment, decide where
> this runs — then prove it twice at $4 before spending $24.
