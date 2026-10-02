# Multi-expert review — the triplicated Appwrite client

**2026-09-29.** Commissioned after two defects spread through three copies of the same module in
one day, one of them Tier 1 and one of them the cause of the first-ever FAO delivery failure.

---

## 1. System Summary

Three independent implementations of an Appwrite storage/metadata client exist on this platform.
`views-pipeline-core/modules/appwrite/` is 4 files and 3,801 lines, dominated by a single
2,925-line `file.py` exposing 27 public methods. `views-faoapi/managers/appwrite/` and
`views-crafdapi/managers/appwrite/` are 10 files each — 2,343 and 2,150 lines — exposing 6 public
methods from a `metadata.py` that is **exactly 560 lines in both**, differing in 66 lines whose
leading entries are **import order**.

All three descend from one ancestor: each carries the comment
`# <-- CHANGED from "FOUND" to "FOUND_BY_HASH"` verbatim. Consumers are split —
`views-postprocessing` reaches pipeline-core's copy through ports
(`unfao/store_port.py:27`, `crafd/store_port.py:27`), while faoapi and crafdapi each use their own.

Two defects propagated through all three in a single day: content-hash deduplication that reports
success having written nothing (C-155, Tier 1, broke the 2026-09-29 FAO delivery), and
`get_latest_file_id` promising "newest by creation timestamp" while returning `files_list[0]` from
an unordered query (#555, live in a guard since August). **Both are fixed in one copy and present
in the other two.** The platform delivers to two external partners, the UN FAO and CRAF'd.

---

## 2. Expert Reviews

### Robert C. Martin — SOLID, boundaries, dependency direction

**Strengths**

- The **ports in views-postprocessing** (`unfao/store_port.py:27`, `crafd/store_port.py:27`) are a
  correct DIP application: the delivery logic depends on a `latest_file_id(filters)` abstraction,
  not on Appwrite. This is why #314 could move its query loop into `delivery/findability.py` and
  take a `resolve` callable — the seam already existed.
- **faoapi and crafdapi decomposed where pipeline-core did not**: 10 files against 4, 6 public
  methods against 27. Whatever else is wrong, the downstream copies applied ISP to a surface the
  upstream never split.

**Weaknesses**

- **SRP is violated at the file level in pipeline-core.** `file.py` at 2,925 lines holds upload,
  download, dedup, metadata search, container provisioning and pagination. It is the file where
  both defects live, and its size is why `Query.limit(1)` at line 1190 and the docstring promise at
  `datastore.py:517` could contradict each other without anyone noticing.
- **The dependency direction is inverted for two of three consumers.** faoapi and crafdapi do not
  depend on an abstraction; they depend on a *copy*. There is no interface to substitute, so LSP
  and DIP are not violated so much as unavailable.
- **The 27-method surface is not segregated.** faoapi needs 6. It took 6 by copying rather than by
  depending on a narrower interface, which is ISP solved with a text editor.

**Improvements**

1. Extract the 6-method surface faoapi and crafdapi actually use into a published interface, and
   have all three depend on it. The evidence that 6 is the right number is that two independent
   teams arrived at it.
2. Split `file.py` along its existing seams — metadata search, storage, provisioning, dedup. The
   defects clustered in one region of a file too large to review as a unit.

---

### Gang of Four — patterns, and patterns misapplied

**Strengths**

- **`OperationResult` is a consistent Result pattern** across all three copies, with `success`,
  `code` and `data`. It is the reason the dedup defect was diagnosable at all — `FOUND_BY_HASH`
  vs `FOUND_BY_NAME` named the branch that failed.
- **The port/adapter pair in views-postprocessing** is a textbook Adapter, and `findability.verify`
  taking a `resolve` callable is a clean Strategy that made the rule testable without Appwrite.

**Weaknesses**

- **This is Copy-Paste-as-Inheritance.** Three subclasses of an abstract client that was never
  written. The `# <-- CHANGED from "FOUND" to "FOUND_BY_HASH"` comment surviving verbatim in all
  three is the marker: it documents a change to an ancestor none of them can reference.
- **No Template Method where one is obvious.** All three implement the same
  query → paginate → filter → select sequence, and all three got the *select* step wrong in the
  same way. That is precisely the shape Template Method exists to prevent — vary the step, share
  the sequence.
- **`check_file_exists_by_hash` is a Strategy with one hard-coded strategy.** Identity is decided
  by content hash alone; name-identity is a second strategy the caller cannot select, which is why
  the fix had to be a new parameter rather than a different policy object.

**Improvements**

1. If the three copies persist, at minimum extract the query→paginate→select sequence as a shared
   Template so the select step cannot diverge silently again.
2. Make artefact identity an explicit policy (`by_hash`, `by_name`, `by_both`) rather than an
   implicit default. #552 hard-codes `by_both`; a partner-visible store may want to state that.

---

### Michael Feathers — testability, seams, legacy characterisation

**Strengths**

- **`test_serving_isolation.py` exists in both faoapi and crafdapi** and is the strongest artefact
  in this system. It computes the transitive import closure from the route entrypoint and
  AST-scans for producer methods, with a positive control asserting the closure reaches >20
  modules and a tripwire test proving the matcher fires on an injected call. That is a seam test
  written by someone who had been burned.
- **`_FakeDatastore` in pipeline-core's publisher tests** is a real in-memory double with
  `fail_on=` injection, allowing torn-run behaviour to be exercised without Appwrite.

**Weaknesses**

- **The three copies have three independent test suites, and none tests the others.** A
  characterisation test proving the three behave identically does not exist, so "they have drifted"
  is knowable only by diffing source. 66 differing lines in a 560-line file is a number nobody
  computed until today.
- **`file.py` at 2,925 lines has no seam at the query boundary.** The dedup defect required
  mutating production code to demonstrate, and pipeline-core reported that a substring mutation
  passed 40 tests because fixtures were too dissimilar — a characterisation gap, not a coverage gap.
- **The fixtures do not resemble production.** Real filenames share
  `rusty_bucket_forecasting_` and differ only in a timestamp; the fixtures differed wildly. Every
  name-comparison test was therefore weaker than it appeared.

**Improvements**

1. Write **one** characterisation suite, run against all three implementations, asserting identical
   behaviour on the shared 6-method surface. It documents the drift and becomes the acceptance test
   for any future consolidation.
2. Move fixtures to production-shaped names. The mutation that survived 40 tests would have died.

---

### Michael T. Nygard — production, failure modes, blast radius

**Strengths**

- **Manifest-last as a commit marker** (ADR-013 §3.2) is correct stability design: consumers ignore
  unmanifested shards, so a killed run leaves no visible state.
- **The fail-safe on ingest** is right — views-faoapi refused the broken run and served nothing
  rather than serving something it could not validate. The system failed *closed* toward a partner.
- **`_classify_storage_presence`** checking both metadata and storage is the kind of
  belt-and-braces that catches half-writes.

**Weaknesses**

- **The blast radius of a fix does not match the blast radius of the defect.** #552 fixes one of
  three copies. An operator reading "uploads can no longer report success having written nothing"
  will reasonably conclude a Tier 1 class is closed. It is closed in 33% of the code.
- **Two external partners, three code paths, no single point of truth about behaviour.** Which
  client performed an upload determines whether it was correct. That is an operational property
  nobody can observe from outside.
- **The remedy is the trigger.** Re-publishing is the documented response to a torn run (C-105,
  C-22), and a re-publish writes new names over identical bytes — so every artefact hits dedup at
  once. The prescribed fix for a failure triggers the total-failure case. This is the single most
  dangerous property in the system.

**Improvements**

1. Add a version-and-provenance line to every delivery log stating *which client implementation and
   version* performed the upload. Until consolidation, that is the only way to answer "was this run
   affected".
2. Make the C-105/C-22 re-publish procedure conditional on the dedup fix being present, or the
   runbook is instructing operators into the failure.

---

### Martin Kleppmann — data, consistency, distributed correctness

**Strengths**

- **Content hashing is recorded in manifests** (`sha256` per shard), so integrity is verifiable
  independently of the store's own bookkeeping.
- **The wire contract separates Hop-A (per-`(run,target)`) from Hop-B (per-run)**, which is a
  reasonable staged-commit design and gives the postprocessor a place to curate.

**Weaknesses**

- **Content-addressed storage is being used as a name-addressed store.** Dedup keys on bytes;
  every consumer resolves by name. These are different identity models and the system holds both
  without reconciling them. The sidecar — run-independent by construction — makes the collision
  *certain*, not probable.
- **"Latest" is defined by list order, and list order is undefined.** `get_latest_file_id`
  (`datastore.py:517`) takes `files_list[0]` from a query with no `order_desc`. This is reading a
  distributed store and assuming an ordering the store never promised. It has been correct by luck
  since August.
- **The metadata document and the storage object can diverge**, and an `update_document` against
  the wrong file wrote one run's provenance onto another's — observed on 2026-09-29 against run-0,
  still unverified because no route exposes metadata documents.

**Improvements**

1. Make identity explicitly `(name, hash)` everywhere. Either is half an identity; the platform has
   been using each as if it were whole, in different places.
2. Never rely on result order without an explicit sort. Sort after the pagination walk — it is free,
   since the walk already materialises all pages.

---

### John Ousterhout — complexity, deep modules, information leakage

**Strengths**

- **The port interface is a deep module**: `latest_file_id(filters) -> id | None` is a tiny
  interface over meaningful work. views-postprocessing's delivery logic knows nothing of Appwrite.
- **`findability.verify(consumer_name, legs, objects, resolve)`** is similarly deep — four
  parameters, no store types, and its refusals are testable without infrastructure.

**Weaknesses**

- **`file.py` is a shallow module of enormous size.** 27 public methods over 2,925 lines, where
  callers must know that `check_file_exists_by_hash` means *by hash only*, that `get_latest_file_id`
  does not sort, and that `search_files_by_metadata` pages but does not order. **That is information
  leakage as a defining feature**: the interface does not carry what the caller must know.
- **The docstring at `datastore.py:517` is worse than absent.** "returns the file ID of the newest
  matching file based on creation timestamp" is a false abstraction — it describes a module deeper
  than the one that exists, and two consumers built on the description rather than the behaviour.
- **Three copies triple the complexity without tripling capability.** Each is a separate thing to
  understand, and understanding one does not transfer, because the drift is unmapped.

**Improvements**

1. Fix the false abstraction first, before the duplication: either make `get_latest_file_id` sort,
   or rename it to what it does. A misleading name in three copies is worse than duplication.
2. When consolidating, publish the **6-method** surface, not the 27. The narrow surface is the deep
   module; the wide one is the leak.

---

### Rich Hickey — simplicity vs ease, complecting

**Strengths**

- **The delivery declaration is data** (`un_fao.py:43`, `targets = (...)`), and the publish filter
  derives from a mapping rather than restating a rule. Deriving beats duplicating.
- **`OperationResult` separates outcome from value**, which keeps "did it work" and "what is it"
  from being complected into a return type.

**Weaknesses**

- **Copying was *easy*; three clients is not *simple*.** The platform chose ease three times and now
  holds three artefacts that must be reasoned about together and cannot be.
- **Identity is complected with storage optimisation.** Dedup is a *performance* concern; naming is
  an *identity* concern. Binding them means a storage optimisation silently changed what an
  artefact *is*. That is the actual root of C-155, not the missing comparison.
- **"Latest" complects ordering with retrieval.** `get_latest_file_id` is a query and a policy in
  one name, so the policy could rot without the query noticing.

**Improvements**

1. Decomplect identity from storage: an artefact is `(name, hash)`. Dedup may then be a pure
   optimisation that cannot change identity, which makes the whole class impossible rather than
   guarded.
2. Prefer one library that is *simple* over three that were *easy*, but only once the incident is
   closed — consolidating mid-incident complects two changes.

---

### Kent Beck — feedback loops, small steps, working software

**Strengths**

- **The feedback loop that found both defects was excellent** — a real delivery, a pre-registered
  probe suite, five reviewing seats, mutation testing. Two Tier-1-class defects in one day is a
  *healthy* discovery rate.
- **#552 and #314 were both improved by review**, and in both cases the author found a deeper defect
  after being pushed one level up. That is the loop working.

**Weaknesses**

- **The feedback loop does not cross repo boundaries.** Each copy has its own tests, its own CI, its
  own green. Nothing runs that would go red because faoapi's copy still has the defect.
- **A big-bang consolidation is exactly the wrong-sized step** for a platform mid-incident with a
  UN delivery pending. Three copies, two external partners, and an unfixed Tier 1 is not the moment
  to unify a client.
- **"It has been fine since August" is absence of feedback, not evidence.** The ordering bug
  produced no signal because the failure mode is a *false alarm* nobody has yet seen.

**Improvements**

1. **Smallest useful step first**: one characterisation test, run against all three, that fails
   today. It converts "they have drifted" from an assertion into a red bar.
2. Fix the two known defects in all three copies before consolidating anything. Three small changes
   with identical content is the step size this platform can afford this week.

---

### The Maintainer — sufficiency verdict

**What is worth acting on now:** the two defects in the two unfixed copies, and the false docstring.
Those are bounded, high-value, and do not require an architectural decision.

**What is not worth acting on now:** consolidation. Eight perspectives above make a strong case that
one library is correct, and I agree — but the platform is mid-incident-recovery with a UN delivery
pending, a Tier 1 defect half-fixed, and two releases outstanding. Consolidating an Appwrite client
across four repos this week would be the second-largest change on the platform whilst the largest is
unfinished.

**Findings I judge not worth acting on:**

- *Split `file.py` along its seams* (Martin, Ousterhout). Correct, and it will still be correct in
  three months. It is a 2,925-line refactor of the module currently carrying an unreleased Tier 1
  fix. Not now.
- *Make identity an explicit policy object* (GoF). This is designing for a second identity model
  nobody has asked for. #552's hard-coded `by_both` is right until something needs otherwise.
- *Template Method for the query sequence* (GoF). If consolidation happens, this is moot. If it does
  not, it is a shared abstraction across three repos that cannot import each other — the worst of
  both.

**Sufficiency verdict: the review is sufficient to decide, and the decision is "yes to one library,
no to starting now."** The value of this document is that it makes the deferral *explicit and
dated* rather than implicit. A duplication nobody has decided about spreads; a duplication with a
recorded decision and a trigger does not.

---

## 3. Key Disagreements Between Experts

**D-1 — Consolidate now, or fix-in-place first?**
Hickey and Martin argue the duplication *is* the defect and everything else is symptom management.
Beck and the Maintainer argue that three small identical fixes is the step size available this week,
and that consolidating mid-incident complects two changes. **Unresolved by evidence; resolved by
timing.** Both are right about different horizons.

**D-2 — Is the false docstring worse than the duplication?**
Ousterhout says a misleading name replicated three times is worse than duplication, because
consumers built on the description rather than the behaviour — two did exactly that. Kleppmann says
the ordering assumption is the real fault regardless of what the docstring says. **Ousterhout is
right about cause, Kleppmann about consequence.**

**D-3 — Does the port design vindicate or indict the architecture?**
Martin and Ousterhout praise views-postprocessing's ports as correct DIP and a deep module. Feathers
notes the ports exist only where views-postprocessing built them, and that faoapi and crafdapi have
no equivalent — so the good design is one consumer's local virtue, not the platform's.
**The architecture is not vindicated by one consumer doing it properly.**

---

## 4. Failure Mode Analysis

**C-A — A fix in one copy is recorded as a platform fix.** *Tier 2.*
Trigger: closing #551/#552 or #555 without naming the other copies. Location: the three
implementations. Already observed in the making — this review exists because it nearly happened.

**C-B — crafdapi is unassessed.** *Tier 2.*
Trigger: the CRAF'd delivery running with a producer path active. crafdapi carries the identical
dedup defect (verbatim comment, hash-only query) and has a `test_serving_isolation.py`, but nobody
has confirmed the tripwire is green or that its producer paths are dead. faoapi's equivalent was
confirmed today; crafdapi's was assumed.

**C-C — The re-publish remedy triggers the total-failure case.** *Tier 1.*
Trigger: an operator following C-105/C-22 after a torn run, on any repo whose dedup is unfixed.
Every artefact hits dedup at once; N successes, nothing servable.

**C-D — Drift is unmeasured and unmeasurable in CI.** *Tier 3.*
Trigger: any change to one copy. 66 differing lines in a 560-line file was computed by hand today
and is not computed anywhere else.

---

## 5. Long-Term Regret Test

**In one year, what will we regret?**

- **Not consolidating** — moderate regret, slow-accruing. A fourth consumer appears, copies the
  nearest client, and inherits whichever defects that copy holds.
- **Consolidating this week** — high regret, fast-accruing. A four-repo refactor landing on top of
  an unreleased Tier 1 fix, with a UN delivery pending, is how an incident becomes an outage.
- **Fixing two copies and forgetting the third** — the most likely regret, and the cheapest to
  prevent. crafdapi is the one nobody is watching.
- **Not recording the decision** — the highest regret per unit of effort. This review costs an hour;
  rediscovering the duplication after the third defect costs another day like today.

**What will we be glad of?** That the tripwire tests existed. faoapi's `test_serving_isolation.py`
is the reason its copy is inert rather than live, and it was written before anybody needed it.

---

## 6. Engineering Recommendation

**One library is the correct end state. Do not start it now.**

**Now (this week, bounded):**

1. **Fix the two defects in faoapi and crafdapi** — the same changes as #552 and #555, three small
   PRs with identical content. Verify crafdapi's isolation tripwire is green rather than assuming it.
2. **Fix the false docstring at `datastore.py:517`** by making it true, not by softening it.
3. **Record the duplication decision** with a named trigger (below), so the deferral is explicit.

**The trigger for consolidation, stated now so it is not a judgement call later:** a **fourth**
consumer appearing, **or** a third defect propagating through the copies, **or** the FAO delivery
reaching a steady monthly cadence — whichever comes first. Any of those makes the duplication cost
exceed the consolidation risk.

**When it happens:** publish the **6-method** surface that faoapi and crafdapi independently
converged on — that number is the strongest evidence in this review about what the interface should
be — not pipeline-core's 27. Begin with the characterisation suite run against all three, which is
useful whether or not consolidation follows.

**What this review does not decide:** whether the Appwrite client should exist at all, given
views-appwrite#339's standing epic to evict Appwrite from the platform core. If that lands, this
consolidates itself. Nobody should build a shared client without checking that epic's status first.
