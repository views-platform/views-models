# ADR-025: A check that cannot see its subject must refuse, not pass

**Status:** **Accepted** (2026-10-26)
**Date:** 2026-10-26
**Deciders:** Simon (maintainer)
**Informed:** All contributors
**Origin:** epic #608, story #609. Generalised from Cluster G of
`reports/technical_risk_register.md` — fourteen registered concerns, root **C-168**.
⚠️ **§0 records which citations resolve** — the register ones do; §4's first exemplar does not.
**Related ADRs:** [ADR-005](005_testing.md) (the `live` category, whose truthful skip this ADR must
not contradict), [ADR-020](020_errors_must_descend.md) (a refusal still has to name the next file),
[ADR-024](024_pod_runner_contract.md) (the runner contract this rule extends)

---

## 0. ⛔ Where this ADR's citations resolve

✅ **The register's entry citations resolve.** `reports/technical_risk_register.md` on `development`
now runs to **C-181** with `Open 93` and carries the `Cluster G` section, so the root **C-168** and
every `C-16x`/`C-17x` member cited below can be followed from the integration branch.

⛔ **Two figures in this ADR do not, and are marked as-written rather than silently left to read as
current.** §1's *"fourteen of the register's ninety open concerns"* is the 2026-10-25 state. The
register now reads **`Open 93`**, and **Cluster G is 33 members**, not fourteen — widened twice on
2026-10-26 (13 → 30 by #610's sweep, → 33 by the `/falsify` pass). Neither *fourteen* nor *ninety*
resolves against any register revision that exists after this. ⚠️ They are kept because §1 is
describing what the strategic pass found, which is what motivated the rule; **the register is the
authority on membership, never this document.** Caught by review of PR #628 — the first draft of
this very section claimed *ninety* could be followed, which is the defect §2.4 forbids, committed
inside the paragraph written to confess it.

⚠️ **Read the next paragraph too** — one code citation is still unresolvable.

**What was wrong, and is kept here rather than quietly deleted.** When this ADR was accepted, the
register on `development` was the 2026-10-19 revision — highest concern **C-166**, `Open 78`, **no
occurrence of "Cluster G"** — while the entries the whole document is built on lived in two places
that were not `development`:

| copy | highest ID | open | what happened to it |
|---|---|---|---|
| `development` (2026-10-19) | C-166 | 78 | superseded |
| branch `docs/572-sigma-hat-version-null` | C-178 | 90 | fully contained in the copy below; still holds `pod_sweep_outputs.sh` |
| `~/views-diagnostics/epic608_wip_20261026/` | C-181 | 93 | ⛔ was **outside git** — landed on `development` by **#610's residual**, verified a strict superset of both (no entry, field or `**Notes**` body lost; `C-68`'s title corrected) |

⛔ **One citation is still unresolvable, and it is §4's first exemplar.**
`tools/podrun/pod_sweep_outputs.sh` and `tests/test_pod_sweep_outputs.py` exist only on the unmerged
branch `docs/572-sigma-hat-version-null` (added at `e71c35cc`) and **do not resolve from
`development`**. Every reference to that script — §2.3, §3.2, §4 — carries the ⚠️ marker for this
reason. ⚠️ That branch also carries its own older register revision; it is a subset of what is on
`development` now, so a merge of it must take `development`'s copy.

**Why this section exists at all.** It was written because a review found that §4 declared the
unmerged branch for the *script* and never noticed the *register* was on the same branch — an ADR
about checks that cannot see their subject, citing a subject that could not be seen. It is kept, now
that the register half is fixed, because the shape is the lesson: ⛔ **the second half of a
declaration is the half that gets forgotten.** It is the document applying §2.4 to itself.

---

> **A note on scope.** This is a rule about **instruments** — tests, guards, gates, sweeps, audits,
> validators, liveness probes, CI steps, and a person answering a question about the repository. It is
> not a rule about application code. Where an instrument is named below it is an *example*: the rule
> is about the shape.

---

## Summary

**The problem, in one breath.** A check whose input is out of scope, unreadable, or absent returns a
*clean* verdict, and that verdict is indistinguishable from a genuine pass. The repository then
believes in protection it does not have.

**The decision, in one breath.** Three verdicts, not two: **PASS**, **FAIL**, and **UNVERIFIED** —
and in any aggregate, **unverified dominates**.

---

## 1. Context

`/review-rr strategic` (2026-10-25) found that **fourteen of the register's ninety open concerns
share one root cause**, recorded as Cluster G with root C-168 (both resolve on `development` since
#610's residual landed — §0):

> A check out of scope returns a clean verdict instead of refusing to answer — the instrument's input
> excludes the thing it is asked about, and the resulting pass is recorded as a result.

They span five years and every instrument class the repository has: a test-suite skip (C-92), a shell
presence check (C-112), a commit gate (C-113), a CI contract gate (C-142), a mutation audit (C-159,
C-164), a pinned citation (C-160), an archive guard (C-161), two filesystem sweeps (C-169, C-174), a
collection step (C-170), a config read (C-175) and an artefact read (C-176). **Every one produced a
confident answer about something it could not see, and in every case the answer was recorded and
acted on.**

What it has cost, measured:

| | |
|---|---|
| Eight models trained, delivered their parquets, and their **weights died with the pods** — `STATUS: OK` throughout (#549) | irrecoverable |
| `violet_visitor`'s 2026-10-12 gate posterior | **past recovery** |
| The 2026-10-25 `heavy_freighter` control run was declared ready to destroy on **2754/2754 files byte-identical by sha256** | five artefact classes existed only on the pod; three rescued minutes from termination |
| The omission probe written to catch exactly that returned **EMPTY**, because `find -xdev` excludes `/workspace` (C-169) | 43 files counted while one directory held **156 draw files / 13 GB** |
| A **healthy 300-lesson run was killed** on a report that a config knob was off (C-68, C-175) | ~$8 of pod time and five hours of re-planning |
| The FAO forecast delivery was **stalled 145 days** and nothing detected it (#320) | — |
| A branch was committed **with no parent** and every gate in the merge ritual passed it (C-113) | `git diff` rc=128 swallowed; empty output read as clean |

⛔ **The 2754/2754 verification is the whole problem in one number.** It was the most thorough check
available, it was correct in its own terms, and it proved nothing — because every probe compared the
laptop against `$OUT`, and **`$OUT` is the runner's own declaration of the deliverable. A deliverable
cannot audit its own scope.**

**And the repository already pays for this in CI, three times.** Two workflows exist *only* because
tests went quiet and the suite stayed green:

- `roster_configs_load.yml:7-11` — the test `importorskip`s `views_hydranet`, so in `run_tests.yml`
  it has **always** skipped: *"two production ensemble members sat unloadable from August to
  September while every test passed"* (#404, #458).
- `runtime_smoke.yml:22-26` — *"it went green for the five weeks in which every baseline model was
  broken."*

Both remedies were a new workflow for the instance. This ADR is the rule for the class.

---

## 2. The decision

### 2.1 The rule

> ⛔ **A check whose input is out of scope must fail, not pass. "Nothing matched" and "I could not
> look" must never render as the same verdict.**

### 2.2 Three verdicts

| verdict | meaning |
|---|---|
| **PASS** | the check read its subject and the subject is sound |
| **FAIL** | the check read its subject and the subject is not sound |
| **UNVERIFIED** | the check could not establish that it read its subject at all |

An instrument with only two outcomes has nowhere to put the third state, so it puts it in PASS. That
is the defect, and adding the third verdict is the fix.

### 2.3 ⛔ Unverified dominates

**An aggregate containing one UNVERIFIED is UNVERIFIED**, however many PASSes it holds, and the
unverified branch is evaluated **before** the pass branch.

`tools/podrun/pod_sweep_outputs.sh` (⚠️ **unmerged — §0, §4**) already orders itself this way and
the ordering is the requirement, not an implementation detail: the `UNVERIFIED` branch at `:144-149`
runs before the outside-count branch at `:152-159` and before the clean exit at `:162-165`, whose
message is scoped deliberately — *"This is the only result on which destroying the host is
justified."*

⚠️ The failure mode this forbids is specific: `max(codes, default=0)`. An empty list of parts is not
a healthy whole.

⛔ **And it is live in this repository, so it is named rather than left abstract:**
`tools/liveness/report.py:69-71` — `worst_exit` returns exactly that, and
`tools/liveness/__main__.py` consumes it. An empty or fully-filtered `SURFACES` tuple therefore
exits **0**: a dashboard that observed nothing, reporting healthy. Converting it is epic story
**S7**, and until then §4's citation of that same file is a citation of an exemplar that contains
this defect five lines below the cited range. Per §5.2 a refusal names the next file; so does a
prohibition.

### 2.4 A positive control, and it must be independent

Before an empty result is allowed to mean anything, the check proves it could see *something*.

⛔ **A self-derived control is not a control.** This is the sharpest lesson in the repository and it
was learned the hard way, `.github/workflows/secret_scan.yml:25-31`:

> A ratio check ALONE cannot detect a shallow clone — verified 2026-07-31 by cloning this repo with
> `--depth 1`: gitleaks reported "1 commits scanned / no leaks found / exit 0", and
> `git rev-list --count --all` ALSO reported 1, so scanned/total = 100% and a self-calibrating ratio
> passes with flying colours. **Both numbers come from the same truncated repository, so they agree
> with each other and lie together. Only the shallow flag is an independent witness.**

So a control must come from outside the thing it is vouching for: a sentinel the check plants itself,
a flag the tool reports about its own state, a floor a human set deliberately, or a subject known by
other means to exist.

### 2.5 ⚠️ What this rule does NOT say — the environment/repository line

**It does not ban skipping, and it does not make absence suspicious.**

ADR-005's `live` category **requires** a skip: a live test *"must `pytest.skip` truthfully when the
environment lacks access — never a false red"*, and *"asserts only invariants, never freshness
values"*. That is correct and this ADR does not disturb it.

> **The line: a verdict about the ENVIRONMENT may skip. A verdict about the REPOSITORY may not.**

- *"No credentials on this machine"* is a fact about the environment. **Skip, with the reason.**
- *"views_r2darts2 is not installed here"* is a fact about the environment. **Skip, with the reason.**
- *"My glob over `models/` matched nothing"* is a claim about the repository. ⛔ **FAIL.** An empty
  census means the instrument broke, not that the repository is empty.

This boundary has already been litigated once in this codebase, and the precedent is
`tools/liveness/report.py:38-43`:

> A `.env` exists but does not carry every required variable (#298). Deliberately **NOT** a truthful
> skip: "nothing is configured" is an honest absence of observation, while "configured, but half of
> it" is a misconfiguration a human must fix, and **collapsing the two is what let this repo observe
> its shelf under a foreign key.**

### 2.6 ⚠️ The corollary — an absence may be the signal

The rule read from the other side, from C-176: **a well-built instrument distinguishes "not measured"
from "measured as zero", and blankness is how it does it.** So an empty field is as likely to be good
engineering as a gap.

`views_hydranet/train/training_engine.py:1663-1673` is the worked case: `fed_grad_max` starts at
`None`, not `0.0`, because *"a 0.0 initialiser wrote a real-looking zero on every production run,
making 'the wire is off' and 'the wire is on and the gradient died' the same three characters in the
CSV."*

⛔ **Before reporting an absence as a defect, read the code that would have written the value.**

### 2.7 ⛔ The boundary with Cluster H, stated once so it is not implemented twice

Cluster H's recommended action (*"key on content plus an mtime compared to this run's start, never on
existence"*) and C-168's prescribed fix (*"an mtime sweep rooted at the run's start time"*) are the
same sentence written twice. The boundary:

> **G asks "can this check see its subject at all?" H asks "is what it sees from *this* run?"**

They **compose**, and a correct sweep needs both: enumerate every mount (G) **and** date every hit
against this run's start, with the epoch computed where the files are (H, C-171).

Neither cluster owns the other. A story fixing the G half cites H for the dating half rather than
re-specifying it, and vice versa.

---

## 3. Rationale

### 3.1 Why a rule and not more guards

Because the lesson demonstrably does not transfer by being written down once. C-113's own note:

> the lesson recurred in a different subsystem **two days after being written down**, which is
> evidence it does not transfer by being documented.

And it had already been learned five months earlier as C-28 (*"CI workflow only checks last script
exit code"*). Three occurrences of one rule, in three subsystems, is the argument for a named,
citable standing rule — which is what `CLAUDE.md` says an ADR is for:

> A closed GitHub issue is not where architectural reasoning survives.

### 3.2 ⛔ Why there is no shared helper, and why that is deliberate

**No library. No registry. No decorator. The mechanism is per-instrument.**

The register reached this conclusion independently — *"the mechanism of the fix is per-instrument but
always the same shape"* — and the evidence supports it. The pattern exists as an *idiom* at nine
sites — six named `test_the_check_is_not_vacuous`, the others
`test_the_prefix_checks_are_not_vacuous`, `test_the_census_is_not_vacuous` and
`test_the_value_pins_are_not_vacuous`; `grep -rn "not_vacuous" tests/` is the census — and as a
*mechanism* at exactly one (`pod_sweep_outputs.sh`'s sentinel, ⚠️ unmerged — §0), and **the nine
differ in the only part that matters**: what
"seeing the subject" means for that subject — a glob count, a directory's existence, a package's
importability, a parsed file's non-emptiness. The sentinel is a fifth distinct shape.

The repository states its own extraction bar twice:

- `tools/liveness/report.py:1-5` — *"Extracted ONLY after six checks demonstrably duplicated these
  pieces (WET-before-DRY, epic #238)"*
- `tests/test_roster_conformance.py:851-862` — *"Extracted at the THIRD occurrence, not the second
  ... by then the shape was observed rather than guessed"*

⚠️ And there is a precedent for deliberately *not* extracting:
`tests/test_roster_conformance.py:884-891` keeps two four-line copies rather than create a cycle —
*"a third module to hold one comprehension is not"* worth it.

**So this ADR standardises the vocabulary and the ordering, and leaves the mechanism to the site.**

### 3.3 Why "unverified dominates" rather than "report everything"

Because the consumer of a verdict is usually a shell `&&`, a CI step, or a person skimming. All three
read the aggregate. A summary that contains the truth somewhere but exits 0 is read as a pass — which
is the defect, relocated rather than fixed.

---

## 4. The exemplars — this ADR generalises working code, it does not propose new design

Copy one of these. Three are on `development` today. ⚠️ **The first is not** — the sweep and its
test live only on the unmerged branch `docs/572-sigma-hat-version-null` (added at `e71c35cc`), so
`tools/podrun/pod_sweep_outputs.sh` and `tests/test_pod_sweep_outputs.py` will not resolve from
`development`. Said here rather than left to a reader who follows the path and finds nothing: this
ADR's own rule is that a citation which cannot be resolved must announce itself instead of reading
as though it resolved.

⚠️ **That branch also held the register revision this ADR is built on**, which a review caught and
#610's residual has since landed on `development` — see §0. The script half is still unmerged. That
it took a review to notice the second half of one declaration is the reason §0 is a section rather
than a footnote.

| exemplar | the independent control |
|---|---|
| `tools/podrun/pod_sweep_outputs.sh:116-127` ⚠️ *(unmerged — see above)* | **Plants a sentinel on each mount with the same `find` predicates as the real sweep** (`-xdev`, `-newermt "@$SINCE"`), so the control exercises the instrument rather than a proxy. PID-suffixed, so concurrent runs cannot cross-verify. Excluded from the real sweep, so it cannot inflate the result. And it distinguishes `cannot-plant-sentinel` (the write failed) from `sentinel-not-found` (**wrote it and could not then see it** — the shape of the incident). Read-only mounts are **not** exempted: *"'I cannot write a sentinel here' is not 'nothing was written here'"* (`:97-99`) |
| `.github/workflows/secret_scan.yml:88-124` | An **independent** witness (`is-shallow-repository`), plus an explicit refusal when the count cannot be read at all: *"the vacuity guard cannot confirm the scan walked the history, so this run proves nothing. Failing rather than passing blind."* |
| `tools/podrun/pod_run_model.sh:528,546` | Counts the **contents**, not the directory: *"the writer creates the directory before it writes into it, so `[ -d ]` alone would pass on exactly the state this exists to catch"* — then `die`s rather than reporting OK for a forecast that is not there |
| `tools/liveness/report.py:18-66` + `__main__.py:45-49` | A three-valued exit vocabulary; `exit_code_for` is a **bare dict subscript**, so an unregistered verdict **raises** instead of defaulting, and each surface classifies *before* printing; and a bare `Exception` becomes `UNREACHABLE`, so **a crash is a refusal, not a pass** |

---

## 5. Implementation notes

### 5.1 The author's own test

⛔ **Ask: "what is the independent witness that this check saw its subject?"** If the answer is a
number the check derived from the same source it is judging, there is no control — see §2.4.

### 5.2 A refusal must still name the next file

Per **ADR-020**, a refusal is an error a human reads, so it names the next thing to open. *"UNVERIFIED"*
alone is not enough; *"UNVERIFIED: the mount table yielded no sweepable mounts"* is.

### 5.3 Floors are literals a human bumps, with their provenance

Where the control is a census floor, the number is a literal with a comment saying what the census
was when the floor was set, and it is set **below** the current count with a stated margin — it
catches collapse, not drift.

**Copy `tests/test_sample_count_matches_declared_metrics.py:91-96`** — floor 40, census 42, and the
margin stated in the assertion message: *"only {n} models declare num_samples; this guard was
written when 42 did. If models were retired, lower the floor deliberately."*
`tests/test_target_scaler_is_declared_with_a_loss.py:74-80` carries the provenance in the same
shape but sets the floor **at** the census it was written against (40 of 40), so it documents the
number without leaving the margin this section asks for;
`tests/test_target_prefix_convention.py` states the margin qualitatively against counts of 190 and
24. Only the `sample_count` site has both.

⚠️ **Four of the nine vacuity sites do it this way, not nine.** Six carry a numeric floor at all,
and three of those are bare literals with no provenance —
`tests/test_datafactory_client_is_declared.py:39-41` (`assert len(IMPORTERS) >= 20, IMPORTERS`),
`tests/test_darts_entity_id_matches_level.py` and
`tests/test_point_forecast_declares_point_metrics.py`. ⛔ An earlier draft of this section named the
first of those as the one to copy, *because it was the shortest* — and it is the shortest precisely
by omitting what this paragraph requires. Bringing those three up to the shape is not filed; it is
small enough to do in passing, and noted here so the next person to touch them knows which shape is
intended.

### 5.4 ⛔ Proving the guard, both directions

A guard added under this ADR is proved by mutation **after committing**, and that is **not
sufficient on its own**:

- **C-159** — verify the mutation *changed the subject* before recording a verdict. A mutation that
  did not land reads as a survivor; one that broke something unrelated reads as caught, *"and the
  repository starts believing in protection it does not have."* Syntax-check the mutated artefact so
  *"the guard fired"* is distinguishable from *"I broke the file"*. Record **UNRUN**, never a verdict.
- **C-164** — *"a mutation proves a guard is not DECORATIVE, and only a real subject proves it is not
  OVER-CONSTRAINED."* A mutation test applies a wrong input; *"it is silent about whether the guard
  also objects to right input, because supplying right input is not a mutation."* So also run the
  guard against a real, legitimate subject. Where none exists yet, that is a **stated limit of the
  proof**, not an omission.

⚠️ `git status --porcelain` before the first mutation: `git checkout` during a revert has silently
discarded an uncommitted fix on this platform **three** times.

### 5.5 ⚠️ This ADR is itself subject to §2.4

A rule that cannot be checked is a rule that rots. ⛔ **No mechanism enforces this ADR**, and that is
stated rather than implied: there is no linter for "did you think about your control". What the epic
does instead is convert the known instances one at a time, each with its own guard, so the ADR's
claims are backed by sites rather than by assertion. A future contributor adding an instrument is
held to this by review, not by CI.

---

## 6. Consequences

### Positive

- An instrument that cannot see its subject becomes **loud at the moment it breaks**, rather than at
  the moment someone happens to know what the answer should have been (C-169 was caught only because
  13 GB *had to be somewhere*).
- The three CI workflows added one-per-incident stop being the remedy pattern.
- "I do not know" becomes a **sanctioned answer**, which removes the pressure that produces the
  confident wrong one.

### Negative

- Instruments get longer. A positive control is extra code on the path of every run, and the design
  preferences warn that *"each guard is then on the path of every later change."* ⛔ Mitigated by
  §3.2: no shared machinery, and a control is added where an instance was **observed**, not
  pre-emptively everywhere.
- Some checks will become noisier in degraded environments. §2.5's line is what keeps that bounded,
  and it will need judgement per site rather than a rule that decides it.
- ⚠️ A floor is a number that goes stale. §5.3 keeps them collapse-detectors rather than
  drift-detectors, but they are still maintenance.

---

## 7. Open questions

1. ✅ **RESOLVED 2026-10-26 (#623), and it is this ADR's worked example of §2.5.**
   `tools/liveness/report.py:26-29` maps `SKIP_NO_CREDENTIALS`, `SKIP_NO_PACKAGE` and
   `VPN_REQUIRED` to exit 0. Under §2.5 that is defensible — they are facts about the environment,
   not about the repository — but exit 0 is also the answer for *healthy*, so a caller cannot tell
   the two apart. **Simon's ruling: the exit code stays 0, and the report must say what it did not
   observe.** The exit code is unchanged so that nothing consuming it starts reporting a fault on a
   credential-less machine, where not-observing is the normal state; the ambiguity is removed for a
   *reader* instead. ⚠️ **Its stated limit, which §2.3 would otherwise require differently:** a
   script still cannot distinguish "all healthy" from "observed nothing" by exit code alone. That is
   accepted deliberately, and gating anything automatically on this dashboard would be a new
   decision. ⛔ **Not yet recorded in the governed contract.** `docs/CICs/LivenessChecks.md` §6
   carries the exit-0 truthful-skip row, but the ruling's operative half — *the report must say what
   it did not observe* — is absent from that file on every branch, verified 2026-10-27. An earlier
   draft of this section claimed it was recorded. **Amending §6 is outstanding and belongs with
   S7**, which is already touching `tools/liveness/`.
2. **ADR-010 has no tier category for irreversible loss of evidence**, which is why C-161, C-168 and
   C-170 sit at Tier 2 beside naming inconsistencies. Reconciling the register's applied tiers with
   ADR-010's written ones is open and is the maintainer's.
3. **Is Cluster G's membership complete?** At least seventeen further entries match the root cause
   and are not tagged G. §2.7's boundary is what stops the answer being "all of them". ⚠️ **#610
   (S2) is CLOSED COMPLETED (2026-10-26 00:55Z), and its result is not on `development`** — the
   membership work was done in a copy outside git and has since been landed on `development` as
   #610's residual (§0), verified a strict superset. **So the widening is now on the integration
   branch: Cluster G is 33 members, not 14** — 13 → 30 by the sweep, → 33 by the `/falsify` pass the
   same day. The counts in §1 and §8 describe the cluster as this ADR was written against it; §0
   marks them as-written.

---

## 8. References

- `reports/technical_risk_register.md` — Cluster G and root **C-168**; members C-92, C-112, C-113,
  C-142, C-159, C-160, C-161, C-164, C-169, C-170, C-174, C-175, C-176. ⛔ **Eight of these thirteen
  resolved on `development` when this ADR was accepted; all thirteen and the root C-168 resolve now
  that #610's residual has landed — see §0**
- Epic **#608**; this ADR is story **#609**; tracker **#624**
- **#549** (weights died with the pods), **#558** (the archive guard counts only the half it
  archives), **#320** (145 days undetected), **#404** / **#458** (unloadable ensemble members),
  **#343** and **C-28** (the exit-code lesson, twice)
- ⚠️ **A figure this ADR deliberately does not cite as measured.** The claim that guards which
  cannot fire were found in *"9 of 18 repositories"* appears in the `falsify` skill's own prose
  (`~/.claude/skills/falsify/SKILL.md:19`) and is attributed there to
  `reports/measurements/2026-08-23_agent_failure_pattern_prevalence.md`. ⛔ **That file does not
  exist in this repository, or anywhere under `views_platform/`** — verified 2026-10-26. The
  register repeats the same citation twice (C-148, C-159). The prevalence claim may well be true;
  it is **unverified from here**, and under §2.4 an unverifiable witness is not a witness. ⚠️ It
  was recorded as a thread on **#610, which has since closed** — so it is restated here, in the
  accepted document, rather than left to a closed issue
- ADR-005 §Test Categories (the `live` amendment), ADR-020, ADR-024
