# Postmortem: epic #342 — making forecast delivery declared, checked and derived

**Date:** 2026-08-05
**Author:** Simon / Claude (prompted by Simon)
**Status:** Epic complete — all seven stories merged to `development`; suite 7630 passed, 0 failed
**Scope:** views-models `deliveries/`, `postprocessors/un_fao/configs/`, `docs/ADRs/017|019|020`, `docs/forecast_delivery_map.md`
**Related:** ADR-017, ADR-019, ADR-020 (all merged #341) · epic #342 · tracker #350 · PRs #351–#357 · views-pipeline-core #398–#400 · register C-113, C-127–C-131

---

## 1. Executive summary

Delivery was real, consequential and undeclared. Which forecast reached the UN was one
hand-typed line inside a file whose own docstring said *"modifying it will not affect the
model"* — a sentence that was false for the life of the file. Whether it uploaded at all
was a second key that **existed only in one person's working tree**, so two identical
checkouts published differently. "Is this in production?" was answered by a field nothing
branched on.

Seven stories replaced that with a declaration (`deliveries/<consumer>.py`), rules that
check it offline, errors that name the next file to open, and a derived production status
that is never stored. **75 tests** added across seven new files; **five** new modules
under `deliveries/`; register grew to **140** entries.

**The finding that matters most is not any single defect. It is that the guards written in
early stories caught real errors in later ones, including errors I would otherwise have
shipped.** That compounding is the thing worth keeping, and §4 is the evidence for it.

The second finding is less flattering and equally useful: **the dominant error class in
this epic was measuring with a pattern too narrow to see what it was measuring** — three
separate instances, one of them committed *inside the test written to prevent the first*.

---

## 2. What shipped

| story | PR | what it decided |
|---|---|---|
| **#343** S1 | #351 | `deliveries/` + the vocabulary + the FAO declaration, as a *characterisation* of what already ran. Parity test proving the declaration described reality before anything depended on it. |
| **#344** S2 | #352 | Every ADR-019 §4 / ADR-017 §5 rule answerable offline: resolution, level correspondence, the reconciliation graph, `reconciled=False` as a hard error, freshness, tier, R1/R2. |
| **#345** S3 | #353 | Errors descend, plus the **static meta-test** that stops the staircase rotting. `locked_door()` for ADR-020 §5. |
| **#346** S4 | #354 | `is_in_production` derived from maturity + delivery edge, never stored. A report that says plainly it observed nothing. |
| **#347** S5 | #355 | The FAO config **derives** its source from the declaration. First behaviour change. |
| **#348** S6 | #356 | `intent` is the arming switch; `wire_upload_enabled` derived from it, and **withheld when the repo disagrees with itself about the region**. |
| **#349** S7 | #357 | The delivery map's claims become executable. Two had already gone stale — during this epic. |

Preceded by **#341**, which split ADR-017 into 017 + 019 + 020 + a living map, and amended
ADR-000 with the containment rule that made the split legitimate.

---

## 3. The finding that matters most: guards compound

Four separate occasions where a mechanism written for one story caught a real defect in a
later one. None of these were found by review-by-reading.

### 3.1 A raise site that *could not* name a file

#345's meta-test enumerates every `raise` in `deliveries/` statically and asserts the
message names the next file to open. On its first run against #344's code: **nine of ten
sites were right, the tenth was not.**

The cause was structural rather than careless. `_check_reconciliation` never received the
`consumer` argument, so it **could not** name `deliveries/<consumer>.py` even in
principle. The signature made the correct message impossible, and nothing anywhere would
have failed.

It had also silently falsified a claim in PR #352's own description — *"every failure
message names the next file to open"* — written in good faith by someone who had not
checked.

ADR-020 §3 predicted this: *"the staircase rots in its second month — someone refactors a
check, the message becomes `KeyError: 'level'`, and nothing fails."* It rotted at one site
in ten, immediately, in code written by someone actively thinking about the rule.

### 3.2 A guard firing exactly where it was designed to

#343 asserted that nothing under `postprocessors/`, `models/`, `ensembles/` or
`monthly_run.sh` referenced `deliveries/`, because that story was **additive**. #347 is the
story that changes it, and the guard failed there on schedule.

It was **inverted, not deleted**: the same fact is still checked, with the opposite
expected value. A guard removed the moment it becomes inconvenient teaches the next person
that guards are negotiable.

### 3.3 The same guard asking a question worth answering

`COMMITTED_META_KEYS` exists to fail when the FAO config gains a committed key, so somebody
has to decide whether the new key belongs in the delivery declaration. It fired twice —
once for `ensemble` (#347), once for `wire_upload_enabled` (#348). Both times the answer
was *"yes, and it is derived"*, which is exactly the conversation the guard was built to
force.

### 3.4 A meta-test catching a module nobody had assigned a rule

#345's third test asserts that no module in `deliveries/` is covered by neither descent
rule. #346 added `status.py`, which raises errors — and the guard, written the previous
day, caught it immediately.

### 3.5 CI catching what a green local suite could not

Two breaks (#355, #356) passed locally and failed in CI, both for the same underlying
reason: **this working tree is not a clean checkout.** `postprocessors/un_fao/configs/`
carries production values that exist only in the maintainer's tree (register **C-110**), so
CI is testing different files.

The technique that fixes it — stash the uncommitted config, run the suite, restore — is now
recorded in memory and was used to verify #349 in both states before pushing.

---

## 4. What the mechanisms caught (evidence, not opinion)

| mechanism | real defects found | would review-by-reading have caught it? |
|---|---|---|
| Static `ast` enumeration of raise sites (#345) | **5** — the unnameable raise site, plus `paused(since=)`, `months()`, `send=[]` giving no corrected form, plus `status.py` unassigned | No. Four of the five were on paths no fixture reached. |
| Clean-checkout simulation | **2** CI breaks diagnosed, 1 pre-empted | No. Both passed locally. |
| Parity test before dependency (#343 → #347) | 0 defects, but it is what made #347 and #348 *safe to attempt* | N/A — this one prevents rather than detects |
| Guards from earlier stories (§3) | **4** | No — all four fired mechanically |
| The `/falsify` round on the ADRs (pre-epic) | **5** (3 hard, 2 soft) | Partly — the internal contradictions, yes; the wrong measurement, no |

### The single most effective technique

**Static enumeration over dynamic triggering.** A meta-test that only inspects errors it
manages to provoke can only ever check the paths someone already thought to write a fixture
for — and the missed path is, by construction, the one nobody thought about. Six lines of
`ast` found what no amount of test-writing would have.

The runner-up is **inverting guards instead of deleting them**. It costs one edit and it is
the difference between a suite that records what a change was allowed to do and a suite
that quietly forgets.

---

## 5. Where I was wrong, and the shape of the errors

Twenty-odd errors, and they fall into seven shapes. The shapes are more useful than the list.

### 5.1 Measuring with a pattern too narrow to see the thing — three times

This is the dominant class, and the third instance is the instructive one.

1. **ADR-017 §2 and the map both stated a measured distribution that was wrong.** Claimed
   *"120 `shadow`… across all 131 sources"*; actual is **117 / 6 / 4 / 1 across 128 files**,
   of 132 source directories. The measurement used a double-quote pattern; **81 of 128
   files write `{'deployment_status': 'shadow'}` with single quotes** and were silently
   reported as absent. Registered **C-127**.
2. **#349's interlock check** used `UPLOAD_ENABLED\s*=\s*False`. The real line is
   `UPLOAD_ENABLED: bool = False`. A present constant read as missing.
   **That is C-127's defect, committed inside the test written to prevent C-127.**
3. **A `str.replace` that silently did nothing**, because I built the `old` string from
   terminal output I had prefixed with `sed 's/^/  /'` — two leading spaces that do not
   exist in the file. The script asserted its match count, so it failed loudly rather than
   corrupting the document. That assertion is the only reason this is a footnote.

**The lesson is not "be careful with regexes."** It is that a measurement over a corpus must
either be validated against a total (`assert files_matched == files_found`) or parsed rather
than pattern-matched. §2's numbers *were* ADR-017's argument; a mis-measurement there is not
cosmetic.

### 5.2 Tests that could pass while proving nothing

- **`git grep` returns 128 with empty stdout** when it cannot run. The `#343` guard read
  that as "nothing references `deliveries/`" and **passed on error** — in the one test whose
  whole job was proving the story was additive. This is a recurrence of **C-113**'s lesson
  *two days after it was written down*, in a different subsystem; registered as an update to
  C-113 rather than a new entry, because the finding is that the lesson did not transfer by
  being documented.
- **`views_postprocessing` imports as a namespace package** here — `__file__` is `None`
  while `__path__` points at a sibling checkout. `pytest.importorskip` alone would have let
  #349 claim it verified an installation it never saw (**C-75**'s failure mode).
- **The parity guard `exec`'d the committed config**, which after #347 needs `__file__`. The
  trap in the timing is worth naming: **the full suite passed *before* the commit**, because
  HEAD still held the old plain-dict version. A green suite immediately before a push is not
  evidence when the thing under test is *"what is committed."*
- **The arming tests asserted a value that depends on which checkout you stand in.** Fixed
  by asserting the *relationship* — if the two configs disagree, the upload must not be armed
  — which is true everywhere. Asserting they agree would have been asserting C-110 is
  resolved.

### 5.3 Design errors, all caught before shipping

- **ADR-019 had no home for `wire_upload_enabled`** — the switch that actually arms the
  delivery. The ADR mentioned it zero times while the map mentioned it twice. As written, the
  design would have lifted the `"ensemble"` line out of the dishonest file and **left the
  on/off switch inside it**. Found by the `/falsify` adequacy probe; registered **C-129**.
- **I nearly invented a second `tier` value.** A first draft added `internal`, meaning "no
  maturity requirement, so a candidate *can* be delivered." ADR-017 says a candidate writes
  nowhere central, and §12 explicitly leaves open *where* a non-production forecast should
  land. I would have answered an open question in a footnote. Reverted; the second value is
  now blocked on §12 with that named as its trigger.
- **My own issue #347 said "remove the `ensemble` key."** It is read one repository away at
  `views_postprocessing/unfao/managers/unfao.py::_resolve_run` (`expected_ensemble=self.configs["ensemble"]`). Removing it raises `KeyError` in a repo
  this epic never touches. Tracing before implementing turned a breaking change into a better
  design: the key survives as an *interface* and dies as a *decision*.
- **#348's first version raised on a region mismatch.** That would have made the config
  unloadable from a clean checkout, breaking runs that never intended to upload — inventing a
  new failure mode to guard an old one. It now disarms and warns, which is what the interlock
  already does when the key is absent.

### 5.4 Dependency direction

`deliveries/coherence.py` imported `load_config_module` from `tests.conftest`. Production
code depending on the test package inverts the dependency and makes `deliveries/` unusable
without pytest installed. Ten duplicated lines instead — and the reason written in the
docstring, because a bare duplicate looks like an oversight.

### 5.5 Latent logic that today's data cannot expose

- **`in_production()` conflated two deliveries.** It checked *"some prod-tier delivery
  exists"* and *"this source is delivered somewhere"* as separate conditions. Indistinguishable
  today because `tier` has one value; **wrong the moment it has two** (C-131).
- **`maturity_of()` recursed into members with no cycle guard**, so a self-containing
  ensemble would give `RecursionError` — the cryptic failure ADR-020 forbids.

Neither could fail on current data. Both would have waited for the change that made them
reachable.

### 5.6 Cross-repo line-number citations rot, and mine already have

Every story in this epic cited the consumer of the `"ensemble"` key as
`views_postprocessing/unfao/managers/unfao.py:140` and `:195`. Writing this postmortem one
day later, those lines are **152** and **207** — the file was edited in
views-postprocessing (`a69d1a0`, *"C-79 and C-83 — two refusals on the live FAO path"*)
between the trace and now.

Nothing broke: the citations were evidence for a design decision, not a dependency. But
they are now wrong in ADR-019, in `config_meta.py`'s docstring, in three test docstrings
and in five commit messages — and a reader who follows them lands on a docstring and
concludes the claim was never true.

**The fix is to cite the symbol, not the line.** `docs/forecast_delivery_map.md` already
does this correctly in places (*"`unfao/managers/unfao.py::_read_forecast_data`"*) and
incorrectly in others (*"`product.py:36`"*, *"`api.py:148,278`"*). The map's own re-check
table is the place this matters most, because its entire purpose is being followable.

Cheap follow-up, not done here: sweep the line-number citations in ADR-019, the FAO config
docstring and the map, and replace them with symbol references. Listed in §8.

### 5.7 Process and communication

- **I skipped `/ship-it`'s second gate on #344**, running only the new test file. CI caught
  what I should have. Every story after that ran the full suite, and #349 ran it twice.
- **I said "running the ritual on it now" and then produced a status table instead**, which
  led directly to *"I don't understand; are we done here?"* The answer was no, and the
  confusion was mine to own.
- **I told the maintainer C-110 covered only `wire_upload_enabled`** and said I would widen
  it. Checking first showed the register already listed `region` and `wire_contract`. The
  claim was wrong; nothing else changed.
- **My own epic acceptance check produced two false positives**, because it used
  `origin/main` as the baseline — 350+ commits behind — and grepped for a docstring phrase
  that appears as a *quotation explaining the past*. Verifying a verification is not paranoia
  at this point.

---

## 6. What worked and should be kept

- **Write the failing test first, and make the first test a *parity* test.** #343's parity
  assertion — the declaration equals the config it will eventually replace — is the single
  reason #347 and #348 were safe to attempt at all.
- **Invert guards, never delete them.**
- **Enumerate statically; do not rely on triggering.**
- **Simulate a clean checkout before pushing**, while C-110 exists.
- **Warn, do not block, for declared transitional violations** — and pin the *warning* with a
  test, so converting it to a hard failure is caught in review rather than discovered when a
  delivery stops.
- **Derive rather than duplicate.** Both `ensemble` and `wire_upload_enabled` now survive as
  interfaces to another repository while ceasing to be decisions here. No cross-repo change
  was needed for either.
- **Stage narrowly when the working tree carries production values.** #347 and #348 committed
  only their own change, via `git hash-object` + `update-index`, leaving C-110's state exactly
  as found. Committing `region: land_gaul` while `config_queryset.py` still said
  `africa_me_legacy` in git would have left two committed files disagreeing about which region
  ships — worse than the split it appeared to fix.
- **Record the reason in the artifact, not only the commit.** Every superseded test says why
  it was superseded, in place. A tautology that reads like coverage is worse than no test, and
  the next person needs to know that without archaeology.

---

## 7. What to change

1. **A measurement over a corpus must validate its own reach.** Assert the matched count
   equals the file count, or parse instead of pattern-match. Three instances in one epic is a
   process gap, not bad luck.
2. **Run the suite in both states before every push** while C-110 exists — not just the
   suite. Two CI breaks would have been caught locally.
3. **Do not write "doing X now" unless X is the next action.** Status and action are
   different messages.
4. **Baseline acceptance checks against the epic's start commit**, never `main`, which is
   350+ commits behind on this repo.
5. **Cite symbols, not line numbers, across repository boundaries.** Mine rotted within a
   day (§5.6). A line number is a claim about a file you do not control.
6. **`Closes #NNN` does not auto-close here**, because merges target `development` and GitHub
   only auto-closes on the default branch. Close by hand; it was missed once.

---

## 8. Open items carried out of this epic

| item | owner | note |
|---|---|---|
| **views-pipeline-core #398/#399/#400** | maintainer | The `deployment_status → maturity` rename and re-homing the dead R2 guard. Blocks nothing here — R1/R2 run as repo tests with ADR-017 §3's mapping applied in memory. Cannot ride the 3.0 release. |
| **C-110 residual** | maintainer | `wire_contract` and `region` are still working-tree only. A clean checkout now **disarms visibly** rather than shipping the wrong region, so the state is safe; committing `config_queryset.py`'s `land_gaul` is a decision about production config. |
| **C-97 is Resolved on a claim that is not true** | maintainer | It says a manifest-addressed run *"is what faoapi serves"*; faoapi reads `unfao_bucket` and that run is in `production_forecasts`. The map records this. Correcting a register entry's status is curation, not a side effect of a test story. |
| **G3–G5** | maintainer | A real `monthly_run.sh`, a real FAO delivery (blocked on the **C-90** decision about `rusty_bucket`), then `main` and `0.3.0`. |
| **Cross-repo line citations are stale** | — | `unfao.py:140`/`:195` are now `:152`/`:207` (§5.6). Wrong in ADR-019, the FAO config docstring, three test docstrings and the map. Sweep them to symbol references; the map matters most, since being followable is its purpose. |
| **#333 `un_crafd`** | — | Can now arrive as a declaration rather than a clone of `postprocessors/un_fao/`, which was ADR-017 §11 Phase 1's stated reason for sequencing this work first. |
| **`monthly_run.sh` as a filter over declarations** | — | C-122. Out of scope here; `frequency` has a reader via #346's report without touching the production ritual. |

## 9. What this epic did not do

It **did not fix #320.** The FAO forecast stream is still stale. Declaring a delivery
honestly is not performing one, and nothing here shipped a forecast. That distinction is
written into the epic description, the report's own output (*"DECLARED, not observed"*), and
this document, because a green epic must not be mistaken for a delivered forecast.

**No delivery was performed at any point.** No file under any of the eight live hydranet
experiment models was changed — verified against the epic's start commit, not `main`.
