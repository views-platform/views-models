# Technical Risk Register — views-models

**Governing ADR:** [ADR-010](../docs/ADRs/010_technical_risk_register.md)  
**Total entries:** 195 (186 concerns + 9 disagreements)  
**Concerns:** Open 95 | Mitigated 32 | Resolved 48 | Accepted 5 | Partially Resolved 1 | Subsumed 1 | Merged 4  
**Concerns by tier:** T1 6 | T2 74 | T3 75 | T4 27 (4 merge stubs carry no tier — C-22, C-67, C-87, C-93)  
**Disagreements:** Open 7 | Resolved 1 | Subsumed 1  
**ID note:** IDs run C-01–C-187 with **C-132 never assigned** — 186 concerns, highest ID 187. Nothing references C-132; the gap is recorded here so no future reader searches for a deleted entry.  
**Last updated:** 2026-10-30 (see below)  
**Last curated:** 2026-10-30 (#648 — **C-185** (a prefix measured through a symlink, reporting the target's facts as the link's, with an `rm -rf` remedy that would have emptied the maintainer's real environment), **C-186** (a delivery declaration recording an upload as never having happened while its own `wandb/` log recorded 111 uploads on 2026-08-14), and **C-187** (an evidence directory holding a module CPython auto-imports) registered; **C-127** given a second mechanism — a count-based guard that cannot identify WHICH instance it counted, and so held a stale label in place through its own subject's correction; Cluster G row counted rather than inherited, 35 → 38, after finding C-182 and C-184 declared membership but were never listed. Four findings from the same reviews were fixed with tests and deliberately NOT registered: a zero-row fetch building an empty mapping, a re-arm gate naming closed blockers, one measurement written in two precisions, and an orphaned percentage.) Previous: 2026-10-27 (#635 — **C-184** registered from the independent guard-mode `/falsify` on #635's commits: 25 mutations, 9 survivors, four fixed inside #635 and each proved by re-running its mutation, the fifth registered because the fix is fleet-wide; Cluster G 35 → 36, both prior figures found stale and corrected against a measured count). Previous: 2026-10-27 (epic #595 S3 — **C-182** registered from the guard-mode `/falsify` on `09a3992b`, Cluster G 33 → 34; **C-183** registered after the maintainer challenged an unchecked memory claim, confirming C-155's carry-forward one tier up). Previous: 2026-10-26 (`/falsify` on "I can terminate the pod" during the #607 arm teardown — C-179, C-180, C-181 registered; C-161 given a third instance; C-169 given a second instance by a second mechanism; Cluster G 30 → 33). Previous: 2026-10-26 (#610, epic #608 — **Cluster G widened 13 → 30 members** after a membership sweep; 6 moved out of Cluster A; C-106 recorded as G's root one layer down, in CI; the G/H boundary written down; C-176's missing `Location` row added; 25 entries given the `---` separator they lacked; 2 Trigger-index rows added for C-92 and C-112; a dangling citation in C-148 and C-159 corrected). Previous: 2026-10-25 (`review-rr strategic`, second full pass — trigger index added, clusters A–H defined, Cluster G given a root, C-54 promoted 4→3, two unqualified cross-repo references fixed). Previous: 2026-07-31 (first full pass — tier recalibration, 4 merges, 6 clusters identified but never defined).

> ✅ **Cluster G is now governed by [ADR-025](../docs/ADRs/025_a_check_that_cannot_see_must_refuse.md)** — *a check that cannot see its subject must refuse, not pass.* Three verdicts (PASS / FAIL / UNVERIFIED), and unverified dominates any aggregate. Worked as epic **#608**, tracker **#624**.

> ⚠️ **The tier definitions this register applies are not the ones [ADR-010](../docs/ADRs/010_technical_risk_register.md) states.** ADR-010 defines Tier 1 as "affects multiple models or external consumers"; this register applies "silent data corruption with no error signal" — which is why C-01 and C-02 were demoted out of Tier 1 on 2026-07-31. ADR-010 lists four statuses; this register uses eight. And **neither tier system has a category for irreversible loss of evidence**, which is why C-161, C-168 and C-170 sit at Tier 2 beside naming inconsistencies. Reconciling the two is open, needs the maintainer's decision, and may re-promote C-01 and C-02.

---

## Trigger index

**Read this before an irreversible or outward-facing action. It is the only part of the register that is indexed by what you are about to do.**

Why it exists: on 2026-10-25 a pod was declared ready to destroy while five artefact classes existed nowhere else. C-161 had been open since 2026-10-13 with the trigger *"Destroying a rented pod after `pod_run_model.sh` reports `STATUS: OK`, for any model whose readout depends on the gate"* — a description of the incident twelve days before it happened. It was correct, specific, and unreachable: nothing connected the action to the entry. Ninety open concerns in ID order is not a thing anyone reads before pressing a button.

Coverage is **Open and Mitigated** concerns — a Mitigated entry still carries a live residual, and C-96 and C-139 re-arm on their own triggers. Resolved entries are excluded. An entry may appear on more than one row.

| Before you… | Read |
|---|---|
| **destroy a pod, container or rented host** | C-161, C-168, C-169, C-170, C-171, C-179, C-181, views-hydranet C-317 |
| **write a monitor, guardian or readiness check against a pod path** | C-171, C-177 |
| **locate a pod run's output, or conclude a run did not happen** | C-174, C-169 |
| **sync or rsync a timestamped run directory by name** | C-179, C-171 |
| **compare two checksum manifests, or accept a count as proof of parity** | C-180, C-179, C-159 |
| **rehearse the pod chain before a paid run** | C-172 |
| **rent or size a pod for a pooled / multi-draw ensemble** | C-183, C-155, C-112 |
| **hand a delivered bundle to a recipient** | C-173, C-110, C-117 |
| **merge a PR** | C-80, C-113, C-142, C-163 |
| **push a branch** | C-178, C-105 |
| **report a config key inert, dead or off — or answer "is \<feature\> on?"** | C-157, C-175, C-68, views-hydranet C-324 |
| **run a guard audit or any by-hand mutation pass** | C-159, C-164, C-182 |
| **write or widen a guard, or read a mutation set's kill rate as coverage** | C-182, C-184, C-127, C-113, C-92 |
| **raise, lower or unify a dependency floor — or edit any `requirements.txt` whose model shares a conda prefix** | C-184, C-116, C-112, C-50, C-185 |
| **add or change a `pytest.importorskip`, or a `skipif` on a census** | C-92, C-106 |
| **decide in a shell script whether a credential or coordinate is available** | C-112, C-154 |
| **delete, recreate or repoint anything under `envs/`** | C-185, C-149, C-184, C-116 |
| **state what a partner has already received — to re-arm a delivery, answer an external party, or assess exposure to a sourcing rule** | C-186, C-126, C-121, C-97 |
| **add a file to a retained artefact or evidence directory** | C-187, C-171, C-161 |
| **cite another repository's code** | C-160 |
| **report an empty column, field or file as missing data** | C-176 |
| **arm a partner delivery** | C-121, C-126, C-139, C-98 |
| **create, rebuild or normalise a conda environment** | C-115, C-116, C-150, C-151 |
| **bump partition boundaries** | C-01, C-96 |
| **re-run an ensemble (`--saved`, `-t`, or after a partial constituent run)** | C-85, C-84, C-45 |
| **edit a config and verify it through a loader** | C-156, C-157 |
| **assemble a variance cell or a set of seed replicates** | C-167, C-166 |
| **add a model directory, or sweep over `models/`** | C-54, C-90 |

---

## Causal clusters

Clusters are a `review-rr` extension; [ADR-010](../docs/ADRs/010_technical_risk_register.md) does not define them, which is why A–F existed for three months as inline tags with no definition anywhere. They are defined here now. An entry may belong to two clusters (the C-90 precedent). **This table is the authority on membership, not a grep for `Cluster X`** — ten entries now carry a *"moved from Cluster A"* note recording their history (C-112, C-113, C-164, C-168 on 2026-10-25; C-114, C-133, C-165, C-166, C-167, C-172 on 2026-10-26), so in each of those the **last** `Cluster X` mention in the entry is the cluster it LEFT. ⛔ A naive grep for the last mention returns **A** for all ten and is wrong; a grep for *any* mention over-reports. Read this table.

| Cluster | Root cause | Root | Members | Recommended action |
|---|---|---|---|---|
| **A** | **Declared-but-unenforced** — a value, key or rule is written down and nothing checks that the system honours it | none named | C-01, C-02, C-15, C-39, C-68, C-85, C-115, C-116, C-117, C-120, C-121, C-163, C-173 (+ D-03, D-04) | **Still needs splitting**, but it is smaller: ⚠️ **six members moved to G on 2026-10-26** — C-114, C-133, C-165, C-166, C-167, C-172 — by the same argument that moved C-112, C-113 and C-164 on 2026-10-25. In each, something *is* checked and the check returns a clean verdict over an input that excludes the subject, which is G, not an absent check. 13 concerns under one name is still not a root: it holds a zsh shebang, a credential scope and a partition boundary |
| **B** | **The delivery machine has no closed loop** — nothing downstream confirms what was produced | none named | C-90, C-97 | Thin. Candidates to absorb: C-98, C-111, C-126 |
| **C** | **Cross-repo dependencies have no released contract** | none named | C-38, C-42, C-50, C-73 | Sound as-is |
| **D** | **Catalog tooling is a script, not a program** | C-81 | C-66, C-81, C-93 (stub) | Sound as-is |
| **E** | **Parity-programme drift** — single-model experiments run against a trio designed for parity | C-71 | C-71, C-72, C-74, C-87 (stub) | Closes when the experiments conclude and parity is restored |
| **F** | **No lifecycle for model directories** — nothing distinguishes production from experiment, scaffold or stand-in | **C-54** | C-54, C-90 | Re-anchored 2026-10-25. Was one member (C-90) and it was not the anchor |
| **G** | **A check out of scope returns a clean verdict instead of refusing to answer** — the instrument's input excludes the thing it is asked about, and the resulting pass is recorded as a result | **C-168** (retrieval) · **C-106** is the same root one layer down, in CI — see below | C-26, C-34, C-92, C-95, C-104, C-109, C-112, C-113, C-114, C-119, C-133, C-135, C-138, C-142, C-146, C-148, C-157, C-159, C-160, C-161, C-164, C-165, C-166, C-167, C-169, C-170, C-172, C-174, C-175, C-176, C-179, C-180, C-181, C-182, C-184, C-185 | **The highest-leverage cluster in the register.** Governed by **[ADR-025](../docs/ADRs/025_a_check_that_cannot_see_must_refuse.md)**; worked as epic **#608**. ⚠️ **Widened 2026-10-26 from 13 members to 30, then to 33** (C-179/C-180/C-181, all three from one teardown audit) — see below for the sub-groups and the reason the flat list is not the unit of work. ⚠️ **Membership counted rather than inherited, 2026-10-30: 38 ids in this row.** C-182 and C-184 each declared `Member of **Cluster G**` in their own entry and were never added here, so the row said 35 while the header said 36 — two different stale numbers, neither of them the truth. Both are now listed, with C-185 |
| **H** | **No identity separates "current" from "left over"** — no artefact, directory or cache carries a run identity, so a stale thing and a fresh thing are the same bytes to a reader | C-14 | C-14, C-85, C-105, C-156, C-171 | Fix is one shape repeated: key on content plus an mtime compared to this run's start, never on existence |

### Cluster G — why it outranks the rest

**Thirty-six entries, spanning five years and every instrument class this repository has.** (Measured 2026-10-27 by counting entries whose `Notes` claim membership — `awk '/^### [CD]-/ {id=$2} /Cluster G\*\*/ && id {print id}' | sort -u`. ⚠️ This prose said *thirty-three* and the `Last curated` line said *thirty-four*; neither matched the file, because each registration updated one of them. **No test asserts this count** — it is the same stale-register-prose class as C-31, C-116, C-121 and C-49, and it is why the number is now stated with its method.) Every one produced a *confident* answer about something it could not see, and in every case the answer was recorded and acted on.

The generalisable rule the cluster yields: **a check whose input is out of scope must fail, not pass.** "Nothing matched" and "I could not look" must never render as the same verdict. The mechanism of the fix is per-instrument but always the same shape — a positive control that the check can see *something* before an empty result is allowed to mean anything.

✅ **The rule is now an ADR: [ADR-025](../docs/ADRs/025_a_check_that_cannot_see_must_refuse.md)** (Accepted 2026-10-26), which defines three verdicts — PASS, FAIL, **UNVERIFIED** — and makes unverified dominate any aggregate. It is worked as epic **#608**, tracker **#624**.

#### ⚠️ Widened twice on 2026-10-26 — 13 → 30 → 33 members

The 2026-10-25 pass tagged the thirteen entries found while looking for the root. A membership sweep (#610) then found **seventeen more that match the same root cause** — **eleven** carrying no cluster tag at all (C-26, C-34, C-95, C-104, C-109, C-119, C-135, C-138, C-146, C-148, C-157) and **six** sitting in Cluster A (C-114, C-133, C-165, C-166, C-167, C-172), taking it to **30**. The `/falsify` pass later the same day added **C-179, C-180 and C-181**, taking it to **33**. The sweep's reasoning is recorded per entry. ⚠️ **The honest consequence is that the 2026-10-25 framing understated this cluster by more than half.**

⚠️ **The breakdown above previously read "twelve … and six", which is eighteen, against a total of seventeen.** Eleven is the number the per-entry tags carry, and they are the authority. Caught by review of PR #628 — an arithmetic slip inside the section whose whole job is making this cluster countable.

#### The sub-groups — because a flat list of thirty is not a unit of work

The members share a *mechanism*, not a *fix*. Grouping them by **instrument class** is what makes the list actionable; the table above remains the authority on membership.

| instrument class | members | what the input excludes |
|---|---|---|
| **the test suite** | C-92, C-104, C-138, C-146, C-148 | the dependency that vanished, the runtime behaviour, the re-export, the constructed object, the reader that is not a comparison |
| **CI and merge gates** | C-113, C-119, C-135, C-142 | the command's exit status, the install's real content, the failing upload, the base of the diff |
| **filters and censuses that narrow silently** | C-26, C-34, C-95, C-109 | the broken model, the model with no requirements, the unmapped feature, everything past row 25 |
| **the pod and retrieval chain** | C-161, C-168 *(root)*, C-169, C-170, C-172, C-174, C-179, C-181 | the other half of the posterior, everything outside `$OUT`, the other mount, the four classes the library writes elsewhere, the riskiest stage, the whole filesystem the repo is not on, the second output directory the sync never entered, the mounts a refusal cannot distinguish from an opaque one |
| **config reads and declarations** | C-112, C-157, C-165, C-166, C-167, C-175 | the exported environment, the consumer, the values behind the key set, every subject but the first, the library version, the file that wins the merge |
| **the audit instruments themselves** | C-159, C-160, C-164, C-176, C-180 | whether the mutation applied, whether the citation says what it claims, legitimate input, the code that would have written the value, whether a hash was computed at all |
| **resolved — kept as precedent of recurrence** | C-114, C-133 | the rejected key, the copy the manager actually reads |

#### ⛔ C-106 is this cluster's root one layer down, and is not a member

**C-106** — *"STRATEGIC ROOT: the test architecture verifies declarations exhaustively but runtime behavior nowhere in CI"* — carries no cluster tag and is the same root cause at the **CI layer**, where C-168 is the same root cause at the **retrieval layer**. C-168's own notes say so. It is recorded as G's parent rather than absorbed, because closing it is a different and larger job than any member's; its residual (*"the mitigation was itself blind for five weeks"*) is live. Worked as **#611**, not closed by #608.

#### ⛔ The boundary with Cluster H — stated once so the same work is not filed twice

H's recommended action (*"key on content plus an mtime compared to this run's start, never on existence"*) and C-168's prescribed fix (*"an mtime sweep rooted at the run's start time"*) are the same sentence written twice. Per ADR-025 §2.7:

> **G asks "can this check see its subject at all?" H asks "is what it sees from *this* run?"**

They **compose**: a correct sweep enumerates every mount (G) **and** dates every hit against this run's start (H). Neither owns the other, and a story fixing one half cites the other rather than re-specifying it.

**This cluster is also why the register itself failed on 2026-10-25**, and that instance is not registered as a concern because it is about this file rather than the code: a register is a check, its input is the operator's attention, and 90 open entries in ID order put the relevant one out of scope. The Trigger index above is the positive control.

### Phantom retired

Cluster G was referenced once, at C-142 ("it is also cluster-G shaped"), and defined nowhere. That reference now resolves.
---

## Concerns

> Status is a field, not a section — Open, Mitigated, Accepted, Resolved, Merged and Subsumed entries are interleaved in ID order (ADR-010 §Concern Format). Merged entries are retained as ID-preserving stubs so cross-references never dangle.

### C-01 — Partition boundary updates require atomic edits to 73 files

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A decision is made to change calibration, validation, or forecasting partition boundaries |
| **Source** | repo-assimilation |
| **Status** | Mitigated |
| **Notes** | `meta/partitions.json` is the single source of truth. `tools/partitions/bump.py` (replaces deleted `scripts/update_partitions.py`) rewrites all 100 files with invariant validation, temporal plausibility (val test end ≤ Dec previous year), post-write verification, atomic writes, and JSONL lockfile with git state. `test_config_partitions.py` enforces consistency via shared parser from `tools.partitions.fileops`. Override mechanism (`# PARTITION_OVERRIDE:`) permits declared deviations — see C-56 for staleness risk. **2026-06-06:** ADR-011 migration procedure still references the deleted `scripts/update_partitions.py` — must be updated to reference `python -m tools.partitions.bump`. See ADR-011. **Tier recalibrated 1 → 3 during review-rr (2026-07-31):** the original Tier 1 was impact-only. `tools/partitions/bump.py` now rewrites all 100 files with invariant validation, temporal plausibility checks, post-write verification, atomic writes and a JSONL lockfile, and C-56 (the override-staleness residual) is Resolved — so no silent-corruption path remains. What survives is annual coordination cost under a validated tool, which is Tier 3. Member of **Cluster A** (declared-but-unenforced). |

---

### C-02 — No static validation of queryset correctness

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A VIEWS database column is renamed or removed, or a queryset references a non-existent column |
| **Source** | repo-assimilation |
| **Status** | Open |
| **Notes** | `config_queryset.py` is the most complex config file (up to 734 lines) with zero test coverage. Failures are runtime-only (data fetch phase). Validation would require access to the VIEWS database schema. **2026-04-22 (test-review):** This gap was the root cause of the bright_starship `dict.publish()` crash — `generate()` returns a plain dict for datafactory models but no test validates return type or shape. Minimum viable test: verify `generate()` exists, returns correct type, and that datafactory descriptors contain required keys (`source`, `zarr_url`, `features`). See C-40 (return type contract mismatch). **Tier recalibrated 1 → 2 during review-rr (2026-07-31):** the failure mode as written is a *loud* runtime crash at the data-fetch phase, not silent corruption — Tier 2 under the register's own tier table. **Re-promote to Tier 1 if** a queryset can reference a wrong-but-*existent* column and train silently on the wrong variable; that variant is not currently claimed by this entry and has never been demonstrated. Member of **Cluster A** (declared-but-unenforced). |

---

### C-03 — Integration tests are manual-only, not in CI

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A model breaks at training time but all CI checks pass |
| **Source** | repo-assimilation |
| **Status** | Open |
| **Notes** | `run_integration_tests.sh` is the only mechanism testing actual model training/evaluation. It runs locally and takes hours. The CI pytest workflow (`run_tests.yml`) only runs fast structural tests. A model can be merged broken. **2026-04-10:** Incident confirms this risk — a `df.applymap()` → `df.map()` change in views-stepshifter (commit `06e73a9`) broke all stepshifter model evaluation. Surfaced only by manual integration test, not by CI. See C-31. |

---

### C-04 — Algorithm label / implementation drift

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | `config_meta["algorithm"]` is changed without updating `main.py` imports, or vice versa |
| **Source** | repo-assimilation |
| **Status** | Mitigated |
| **Notes** | `test_algorithm_coherence.py::TestAlgorithmManagerCoherence` validates that `config_meta["algorithm"]` belongs to the correct package family and that the package matches `main.py` imports. Uses a hardcoded `ALGORITHM_TO_PACKAGE` mapping that must be updated when new algorithms are added. |

---

### C-05 — Incomplete hyperparameter validation for non-DARTS models

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A stepshifter or baseline model is created with missing hyperparameters |
| **Source** | repo-assimilation |
| **Status** | Open (cross-repo, pending releases) |
| **Notes** | **Baseline: done.** `views-baseline` has a `ReproducibilityGate` (ADR-014, CIC, 13 tests) on `feature/reproducibilitygate` — covers all 5 algorithms with `CORE_GENOME` + `ALGORITHM_GENOMES`, runtime enforcement in `BaselineForecastingModelManager`, importable contract. **Stepshifter: done.** `views-stepshifter` has a `ReproducibilityGate` (ADR-001, CIC, 17 tests) on `feature/reproducibilitygate` — covers all 5 algorithms with `CORE_GENOME` + `ALGORITHM_GENOMES` (split into `parameter_keys`/`config_keys` for nested params), runtime enforcement in `StepshifterManager._train_model_artifact()`, importable contract. **Remaining:** Both branches pending merge and package release. Once released, views-models can add validation tests following the `test_darts_reproducibility.py` pattern. All three algorithm packages (r2darts2, stepshifter, baseline) will then expose canonical HP contracts. |

---

### C-06 — `config_queryset.py` has unique external dependencies among config files

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Tooling or tests attempt to load `config_queryset.py` without `viewser` and `views_pipeline_core` installed |
| **Source** | repo-assimilation |
| **Status** | Accepted |
| **Notes** | Accepted as intentional deviation per ADR-002. The `viewser` DSL is essential for queryset definition. Testing gap addressed separately via C-02. |

---

### C-07 — Scaffold builder is untested

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A `views_pipeline_core` template update changes scaffold output, causing newly created models to fail existing tests |
| **Source** | repo-assimilation |
| **Status** | Mitigated |
| **Notes** | `test_scaffold_builders.py` added (2026-04-06) with 7 AST-based tests verifying injection seams and 2 functional tests (skipped without `views_pipeline_core`). Injection seams (`input_fn`, `get_version_fn`, `pipeline_config`) allow mocked testing of `build_model_scripts()`. Remaining gap: no test validates that generated scaffold output satisfies structural tests. |

---

### C-08 — `requirements.txt` / `main.py` coherence is unvalidated

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | A model's `requirements.txt` specifies one algorithm package but `main.py` imports a different one |
| **Source** | repo-assimilation |
| **Status** | Mitigated |
| **Notes** | `test_algorithm_coherence.py::TestRequirementsCoherence` validates that `requirements.txt` package name (normalized hyphens to underscores) matches the package imported in `main.py`. **Scope limit found 2026-08-02 (expert-code-review):** that check covers only the *algorithm* package. It does not detect a file declaring an **additional** dependency the model never imports. Two instances existed — `ensembles/skinny_love` and `ensembles/white_mustang` both declared `views-frames>=1.7.0,<2.0.0`, and the sole mention of `views_frames` in either directory was that line. Removed in PR #325; neither environment had the package installed and skinny_love had completed a run without it. The general rule — **declare what you import** — is unenforced in the extra-dependency direction, and closing that is part of the proposed requirements-hygiene test (**D-06**). Cross-refs: **C-116**, **D-06**. |

---

### C-09 — Two placeholder models with no implementation

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | Tooling or documentation includes `thousand_miles` or `thrift_shop` as active models |
| **Source** | repo-assimilation |
| **Status** | Resolved |
| **Notes** | Both directories deleted (2026-04-06). |

---

### C-10 — Conda environments committed to repository tree

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | A contributor mistakes `envs/` contents for tracked repository state |
| **Source** | repo-assimilation |
| **Status** | Accepted |
| **Notes** | Accepted. `envs/` contents are gitignored. The directories exist as local convenience for contributors. |

---

### C-11 — Single deployed model limits deployment path testing

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | A second model moves to `deployed` status and encounters untested deployment-path behavior |
| **Source** | repo-assimilation |
| **Status** | Accepted |
| **Notes** | Accepted as current deployment posture. Deployment gating tested via `test_config_completeness.py::test_deployment_status_is_valid`. Additional models will be deployed as they mature. |

---

### C-12 — Global warning suppression in all model entrypoints

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A numerical instability, deprecation, or data quality issue produces a Python warning that is silently swallowed |
| **Source** | expert-code-review (Martin, Nygard, Hickey) |
| **Status** | Resolved |
| **Notes** | `warnings.filterwarnings("ignore")` removed from all 74 `main.py` files (2026-04-06). Enforcement test added: `test_cli_pattern.py::test_no_global_warning_suppression` (AST-based, parametrized across all models and ensembles). |

---

### C-13 — No prediction quality validation before ensemble aggregation

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A constituent is added to a deployed ensemble's `config_modelset.py`, or an existing constituent's loss / scaler / sample count changes — verify a NaN/Inf/range gate runs before aggregation (C-72 is the realized instance: 46–63% `Inf` cells reached a deployed ensemble's input) |
| **Source** | expert-code-review (Nygard, Kleppmann) |
| **Status** | Open |
| **Notes** | `white_mustang` (deployed ensemble) aggregates via median. No NaN/Inf check or range validation occurs before aggregation. If multiple constituent models produce garbage, the ensemble output degrades silently. Downstream consumers (UN FAO API) receive degraded data. |

---

### C-14 — Training artifacts have no run identity, so they silently overwrite (concurrent *and* sequential)

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Two training runs for the same model execute simultaneously against the same `artifacts/` directory, **or** a model is re-trained and the previous artifact set is replaced in place — in both cases verify whether any prior artifact was needed for reproduction before the run |
| **Source** | expert-code-review (Kleppmann); merged with C-22 during review-rr (2026-07-31) |
| **Status** | Open |
| **Location** | `models/*/artifacts/`, `models/*/wandb/`; DARTS `force_reset: true` in `config_hyperparameters.py` |
| **Notes** | Artifacts have no run ID or timestamp in filenames. Concurrent case: the second writer silently overwrites the first — low probability, but it destroys reproducibility when it occurs. Sequential case: re-running a model overwrites the previous artifacts with no versioning or deduplication; `force_reset: true` in DARTS hyperparameters acknowledges this but does not solve it. W&B logs exist but are not cross-referenced with artifact files, so there is no way to ask "which artifact produced this logged run?". **Merged with C-22 during review-rr (2026-07-31)** — one root cause (no run identity on artifacts), two trigger paths; separate entries invited fixing one and thinking the class was closed. **Tier recalibrated 4 → 3:** silent loss of reproducible state is not a Tier-4 code-quality observation, and the merged partner C-22 was already Tier 3. See also C-85 (the *consumer* side of the same missing identity: the ensemble resolves a cached prediction by artifact timestamp and cannot tell stale from current), C-110 (config reproducibility). Member of **Cluster H** (no identity separates current from left-over). |

---

### C-15 — Zero CIC failure mode test coverage

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A CIC's failure-modes table gains or changes a row — verify `tests/test_failure_modes.py` covers the new/changed mode in the same PR |
| **Source** | test-review (Nygard) |
| **Status** | Mitigated |
| **Notes** | `test_failure_modes.py` expanded from 4 to 9 tests (2026-04-06). New tests cover: empty config files, import errors, runtime errors, integration test runner exit codes. Remaining gap: no tests for scaffold builder `FileExistsError`, no tests for ensemble aggregation failure. 9 of 21 CIC failure modes now covered. **2026-05-20 (test expansion):** `test_failure_modes.py` expanded to ~30 tests with new red-team classes: `TestPartitionBoundaryValidation` (steps=0/−1/default across all models), `TestEnsembleConstituentIntegrity` (config loadability, partition alignment, malformed model lists), `TestMalformedQuerysetDescriptor` (missing keys, None return, circular import). Scaffold builder `FileNotFoundError` now tested in `test_scaffold_builders.py`. Estimated 15 of 21 CIC failure modes covered. **Tier recalibrated 1 → 2 during review-rr (2026-07-31):** this is a test-coverage gap, not a demonstrated silent-corruption path, and both its peers — C-16 (zero direct unit tests on CIC classes) and C-23 (beige-heavy suite) — sit at Tier 2. Holding it at Tier 1 above its own siblings was a peer inconsistency, not a severity judgment. Trigger also rewritten from symptomatic ("a failure mode occurs in production") to actionable. Member of **Cluster A** (declared-but-unenforced). |

---

### C-16 — Zero direct unit tests for any CIC class

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A PR changes a CIC-governed class's public method signature, return shape, or exception behaviour — verify a direct unit test exercises the changed method (not just its downstream output) |
| **Source** | test-review (Beck, Feathers) |
| **Status** | Mitigated |
| **Notes** | All 5 CIC-documented classes (`ModelScaffoldBuilder`, `EnsembleScaffoldBuilder`, `PackageScaffoldBuilder`, `CatalogExtractor`, `IntegrationTestRunner`) have zero direct unit tests. Tests validate their *outputs* (model directory structure) but never instantiate or exercise the classes. 33 CIC guarantees total, only 2 directly tested (6%), 6 indirectly tested (18%), 25 untested (76%). **2026-05-20 (test expansion):** Direct functional tests added for `ModelScaffoldBuilder` (5 tests: dir creation, README generation, subdirs, gitkeep, missing-dir error), `EnsembleScaffoldBuilder` (3 tests: inheritance, dir creation, missing-dir error), `PackageScaffoldBuilder` (8 AST-based tests: class/method existence, create+validate call chain, exception propagation, name validation), `CatalogExtractor` (8 tests: `replace_table_in_section` edge cases, `generate_markdown_table` structure, `create_link` format), `IntegrationTestRunner` (5 tests: help exit 0, nonexistent model warning, unknown flag error). CIC guarantee coverage improved from 6% to ~45%. Remaining gap: runtime behavioral tests for scaffold output satisfying structural tests, ensemble aggregation failure modes. |

---

### C-17 — Tooling scripts are untested "edit and pray" zones

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A developer modifies `create_catalogs.py`, `update_readme.py`, or `generate_features_catalog.py` and introduces a regression |
| **Source** | test-review (Feathers) |
| **Status** | Mitigated |
| **Notes** | 16 characterization tests added in `test_tooling_scripts.py` (2026-04-06) covering: `replace_table_in_section`, `generate_markdown_table`, `generate_repo_structure`, "Created on" regex, and Column extraction regex. Scripts cannot be imported directly (top-level `views_pipeline_core` imports) so pure function logic is duplicated in tests. Remaining gap: orchestration logic (`__main__` blocks, `extract_models()`) untestable without `views_pipeline_core` runtime. |

---

### C-18 — `build_model_scaffold.py` I/O coupling prevents testability

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Any attempt to write automated tests for `ModelScaffoldBuilder` or `EnsembleScaffoldBuilder` |
| **Source** | test-review (Beck), expert-code-review (Martin, Ousterhout) |
| **Status** | Resolved |
| **Notes** | `build_model_scripts()` now accepts optional `input_fn` and `get_version_fn` keyword arguments (2026-04-06). Defaults to `input()` and `PackageManager.get_latest_release_version_from_github()` — backward compatible. `EnsembleScaffoldBuilder.build_model_scripts()` accepts optional `pipeline_config`. Tests pass mock callables to avoid stdin/network. Also fixed `== False` to `not` in package validation. CICs updated. |

---

### C-19 — `create_catalogs.py` has no transactional file write safety

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | `create_catalogs.py` crashes between reading and writing `README.md` |
| **Source** | test-review (Feathers), expert-code-review (Martin) |
| **Status** | Resolved |
| **Notes** | `update_readme_with_tables()` now writes to a `NamedTemporaryFile` in the same directory, then calls `os.replace()` for an atomic rename (2026-04-06). A crash mid-write leaves only the temp file; the original README is untouched. |

---

### C-20 — No timeout or circuit breaker in data fetch path

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | The VIEWS database is slow or unreachable during model training |
| **Source** | expert-code-review (Nygard) |
| **Status** | Open (cross-repo) |
| **Notes** | Models fetch data via `viewser.Queryset.publish()` with no timeout, retry limit, or fallback. A database outage hangs every model indefinitely during normal operation. The only timeout is the external `timeout` command in `run_integration_tests.sh` (1800s), which only applies during integration testing. **Cross-repo location (verified 2026-04-11):** `views-models` only *defines* querysets in each model's `config_queryset.py`. The actual `publish()` calls live in `views-pipeline-core/views_pipeline_core/modules/dataloaders/dataloaders.py:1027,1052` (`get_data()` and the no-drift backup path) plus two metadata-cache call sites in `handlers.py:1691,2124`. Fix must be implemented in `views-pipeline-core` — either by passing a `timeout` parameter to `publish()` (if `viewser.Queryset` accepts one) or by wrapping `get_data()` in a `concurrent.futures` timeout context. Escalate as a `views-pipeline-core` task. |

---

### C-21 — Partition boundary semantics undocumented

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A new contributor or auditor asks "why 121? why 444?" and finds no answer |
| **Source** | expert-code-review (Kleppmann) |
| **Status** | Resolved |
| **Notes** | ADR-011 documents ViewsMonth-to-date mapping, split point rationale, invariants, override mechanism, and migration procedure (2026-04-06). `meta/partitions.json` serves as the canonical reference. |

---

### C-22 — No idempotency guarantee in model training artifacts *(merged into C-14)*

| Field | Value |
|---|---|
| **Status** | **Merged into C-14** (review-rr, 2026-07-31) |
| **Notes** | ID retained as a stub so existing cross-references resolve. C-22 (sequential re-run overwrites artifacts without versioning) and C-14 (concurrent runs overwrite each other) were two trigger paths on one root cause — **training artifacts carry no run identity** — with the same fix. Tracked together at C-14, Tier 3. |

---

### C-23 — Test suite is overwhelmingly beige; red coverage is low

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A model family with no existing red coverage is added (anything beyond the distributional-baseline set the runtime smoke covers) — verify at least one red test exercises its runtime failure path before merge |
| **Source** | test-review (category distribution analysis) |
| **Status** | Mitigated |
| **Notes** | Red coverage improved from 4 to 9 tests (2026-04-06). New tests cover config loading edge cases and integration test runner failure modes. Distribution still heavily beige (~64%) but red category is no longer negligible. Further improvement requires testing scaffold builder and ensemble aggregation failure modes. **2026-05-20 (test expansion):** ADR-005 pytest markers (`@pytest.mark.red/beige/green`) added to all test files and registered in `pyproject.toml`. Red tests expanded to 285 (from 9): partition boundary validation, ensemble constituent integrity checks, malformed queryset descriptors, integration runner CIC coverage. Distribution: 285 red (7%), 2726 beige (67%), 1038 green (25%), 34 unmarked (1%). Suite total: 3775 passed, 308 skipped. |

---

### C-24 — DARTS model `main.py` duplicates manager instantiation

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | A DARTS model's manager constructor signature changes and only one of the two instantiations is updated |
| **Source** | expert-code-review (Martin) |
| **Status** | Resolved |
| **Notes** | All 15 DARTS models and `purple_alien` (HydraNet) refactored to single-instantiation pattern (2026-04-06). Manager is now assigned to local variable before the `args.sweep` branch, matching the stepshifter/baseline convention. |

---

### C-25 — `white_mustang` ensemble uses deprecated CLI import pattern

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | `views_pipeline_core` removes the deprecated `cli.utils` module |
| **Source** | expert-code-review (Martin) |
| **Status** | Resolved |
| **Notes** | `white_mustang/main.py` rewritten to match `cruel_summer` pattern (2026-04-06): uses `ForecastingModelArgs`, no `wandb.login()`, no `LoggingManager`. `test_cli_pattern.py` extended to cover all ensembles via `any_model_dir` fixture, plus new `test_no_global_warning_suppression` test. |

---

### C-26 — `IntegrationTestRunner` `--level` filter silently excludes broken models

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A model's `config_meta.py` has a syntax error; `--level cm` filtering silently skips it |
| **Source** | test-review (Leveson) |
| **Status** | Resolved |
| **Notes** | Fixed in `run_integration_tests.sh:109-153` (2026-04-11). The `--level` filter loop now captures Python stderr to a temp file, checks the subprocess exit code, and on failure: (1) prints `ERROR classifying <model>: config_meta.py failed to load` plus the last line of the traceback to stderr, (2) collects the model in a `CLASSIFICATION_ERRORS` array, (3) **fails fast with `exit 2`** before running any integration tests, listing every unclassifiable model. Manually verified with a synthetic broken `config_meta.py` (`SyntaxError: '(' was never closed`) — script aborts at exit 2 with the model name and traceback line surfaced. Real models still classify cleanly with no regression. The `--library` filter (lines 128-137) uses `grep -q` on `requirements.txt` and does not have the same silent-failure mode. **Cluster G** (added 2026-10-26, #610) — the `--level` filter removes the broken model from the run's scope and the run then passes. The excluded subject is the one the filter was asked about. Resolved, and kept as precedent: this is the cluster's mechanism on record five months before C-168 named it. |

---

### C-27 — Missing `requirements.txt` for `rude_boy` ensemble

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | Dependency tooling or tests assume all ensembles have a `requirements.txt` |
| **Source** | tech-debt-cleanup |
| **Status** | Resolved |
| **Notes** | `ensembles/rude_boy/` was the only ensemble missing `requirements.txt`. Created with `views-pipeline-core>=2.0.0,<3.0.0` matching all other ensembles (2026-04-06). |

---

### C-28 — CI workflow only checks last script exit code

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | `create_catalogs.py` fails but `update_readme.py` succeeds; CI auto-commits corrupted output |
| **Source** | tech-debt-cleanup |
| **Status** | Resolved |
| **Notes** | `.github/workflows/update_catalogs.yml` used `$?` which only captured `update_readme.py` exit code. A `create_catalogs.py` crash was silently ignored. Fixed by adding `set -e` to the run block and removing the redundant `$?` check (2026-04-06). Also removed stale `create_catalogs_01` test branch from triggers. |

---

### C-29 — Dead root-level `config_partitions.py` in `rude_boy` with stale boundaries

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A contributor or tool reads the wrong `config_partitions.py` and assumes `(121, 396)` boundaries are correct |
| **Source** | tech-debt-cleanup (C-01 investigation) |
| **Status** | Resolved |
| **Notes** | `ensembles/rude_boy/config_partitions.py` (root level) had boundaries `(121, 396)/(397, 444)` — 4.5 years shorter calibration window than standard. Framework always loads from `configs/` (which had correct values). Root file was dead code from copy-paste chain (Dylan Aug 2025 → xiaolongsun revert Oct 2025). Deleted (2026-04-06). |

---

### C-30 — `ucdp_extractor` had non-standard partition boundaries from copy-paste error

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | `ucdp_extractor` runs with boundaries `(121, 396)` and forecasting offset `-2`, training on a 4.5-year shorter window than all other models |
| **Source** | tech-debt-cleanup (C-01 investigation) |
| **Status** | Resolved |
| **Notes** | `extractors/ucdp_extractor/configs/config_partitions.py` used `(121, 396)/(397, 444)` boundaries and offset `-2`. Root cause: smellycloud (Nov 2025, commit `901ec1e`) copied from `rude_boy`'s deviant root file instead of using the standard template. Extractor was in `shadow` status, excluded from all CI/testing, so the deviation was never caught. Fixed to standard values (2026-04-06). Test coverage extended to include extractors and postprocessors. |

---

### C-31 — Upstream algorithm package API changes break views-models silently

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A views-stepshifter, views-r2darts2, views-baseline, or views-hydranet maintainer "modernizes" a deprecated API call (e.g., pandas, numpy, sklearn) and merges to development without verifying the views-models environment supports the new API |
| **Source** | incident response (2026-04-10) |
| **Status** | Mitigated |
| **Notes** | views-models has no contract test that validates upstream packages still work in its installation environment. **Concrete incident:** views-stepshifter commit `06e73a9` (`chore: clean tech debt`) changed `df.applymap()` → `df.map()` claiming "deprecated API fix (pandas 2.0+)". `DataFrame.map()` was actually only added in **pandas 2.1.0**, and the production environment runs **pandas 1.5.3**. All stepshifter model evaluation broke at the `_get_standardized_df` boundary. Caught only by a manual integration test for `bittersweet_symphony`. Fix: revert to `applymap()` (works in all versions). The deeper problem: **views-models is installed against a frozen environment, but its dependencies are continuously developed against newer environments.** A boundary contract test (e.g., a smoke test that imports the manager and runs a 1-step prediction on a tiny synthetic dataset) would catch this in CI. Related: C-03 (no integration tests in CI), C-08 (requirements coherence — but that's package name, not API surface). |

---

### C-32 — Scaffold builder does not persist empty standard directories in Git

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A new model is scaffolded via `build_model_scaffold.py` and committed before all standard subdirectories contain files, or an existing model with latent gaps is cloned to a fresh server environment |
| **Source** | manual (2026-04-11) |
| **Status** | Mitigated |
| **Notes** | **Original framing was incomplete.** `update_gitkeep_empty_directories()` already existed at `build_model_scaffold.py:283` and was already called in `__main__`. Investigation on 2026-04-11 revealed the actual root cause: `.gitignore` line 10 (`logs/`) — a repo-wide rule for "Integration test logs" — silently swallowed `models/*/logs/.gitkeep` files even when the scaffold created them. The ranger_* hotfix worked only because the .gitkeep files were force-added (`git add -f`). Downstream managers crashed with `TypeError: unsupported operand type(s) for /: 'NoneType' and 'str'` from `ModelPathManager` path resolution. **Mitigation v1 (2026-04-11 morning):** (1) `.gitignore` changed from `logs/` to `logs/*` + `!logs/.gitkeep` to allow the directory placeholder through while preserving the "ignore log file contents" intent; (2) `build_model_directory()` now creates `.gitkeep` inline immediately after each `subdir.mkdir()` call, so the invariant holds from the moment the directory is created; (3) `update_gitkeep_empty_directories(delete_gitkeep=False)` default flipped — the previous `True` default removed `.gitkeep` from "non-empty" dirs, but gitignored data files (`*.parquet`, `*.pkl`) count as non-empty, so the deletion behavior was a latent footgun that re-introduced the bug; (4) backfilled `logs/.gitkeep` for 4 affected models: `old_money`, `orange_pasta`, `wildest_dream`, `yellow_pikachu`. **Regression recurrence (2026-04-11 evening, commit `cd668ea`):** v1 backfill covered only 4 of 37 affected models. A fresh clone on `/home/simmaa/` running `models/invisible_string/main.py` crashed at `ModelPathManager._initialize_model_specific_directories` (`views-pipeline-core/.../model_path.py:458`) with `FileNotFoundError` on `notebooks/`. Scan revealed **9 models** missing tracked `notebooks/.gitkeep` and **28 models** missing tracked `logs/.gitkeep` — all latent failures on any non-dev checkout. The ranger-incident hotfix patched only the four models that had already been reported; every other pre-existing model remained un-backfilled. **Mitigation v2 (commit `cd668ea`):** backfilled the remaining 37 `.gitkeep` placeholders; additionally tightened `test_model_structure.py` to close the C-33 gap (see that entry). Related to C-07 (scaffold builder testing gap), C-33 (test contract now git-index based). |

---

### C-33 — No CI gate for model directory completeness

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A PR adds or modifies a model such that one of the standard subdirectories (`artifacts/`, `data/raw/`, `data/generated/`, `logs/`) is absent on fresh clone, and the PR merges without the hollow state being flagged |
| **Source** | manual (2026-04-11) |
| **Status** | Resolved |
| **Notes** | **v1 test (2026-04-11 morning):** `TestModelDirectoryStructure` added to `tests/test_model_structure.py`. The class uses the existing `model_dir` fixture (`tests/conftest.py:72`, parametrized over `ALL_MODEL_DIRS`) and asserted every model contained four runtime-critical subdirectories: `artifacts/`, `data/raw/`, `data/generated/`, `logs/`. **Regressed (2026-04-11 evening):** the v1 test had two structural gaps that let C-32 recur unnoticed. (1) `REQUIRED_SUBDIRS` omitted `notebooks/` and `reports/` even though `ModelPathManager._initialize_directories` validates both at runtime (`views-pipeline-core/.../model_path.py:442,458`); a model missing either directory would pass the test and crash on first instantiation. (2) The check used `path.is_dir()` on the local filesystem, so any developer who had ever run a model locally would see the test pass regardless of whether the directory was tracked in git — the exact failure mode C-33 was meant to prevent (fresh-clone absence). C-32's `/home/simmaa/` recurrence was a direct consequence: `invisible_string` passed C-33 locally but had no tracked `notebooks/.gitkeep`. **v2 test (commit `cd668ea`):** `REQUIRED_SUBDIRS` extended to the full set `[artifacts, data/raw, data/generated, data/processed, logs, notebooks, reports]` — parity with `ModelPathManager` runtime validation. The assertion replaced `path.is_dir()` with a `git ls-files` probe via a helper `_git_tracks_path()`, so "pass" means "tracked in the git index" — fresh-clone state, not working-tree state. Coverage now 74 models × 7 subdirs = 518 tracked-path assertions; full suite 3243 passing. See also C-32 (now re-mitigated with 37 backfilled .gitkeeps), C-07 (scaffold builder testing), C-16 (CIC class testing gaps). **2026-06-26 (postprocessor gap — PR #210):** the v2 contract covered `models/` only (`model_dir` fixture over `ALL_MODEL_DIRS`); **`postprocessors/` was uncovered**. `un_fao` — the only postprocessor, run through `PostprocessorPathManager` (same `model_path.py` directory validation) — shipped with **7 missing scaffold dirs** (only `configs/`+`logs/` existed) and crashed at `_initialize_directories` during the vpp#24 `africa_me_legacy` smoke test, before any Appwrite/datafactory work. The contract's filesystem-vs-index lesson held but its *scope* didn't include postprocessors. **Fix (PR #210):** backfilled the 7 `.gitkeep`s (incl. `logs/.gitkeep`, which the `logs/*` gitignore had swallowed — same footgun as C-32) and extended the contract with `TestPostprocessorDirectoryStructure` over a new `postprocessor_dir` fixture (`ALL_POSTPROCESSOR_DIRS`), so the same 7-subdir tracked-path assertion now guards postprocessors. `apis/` (different manager) intentionally out of scope. |

---

### C-34 — `--library` filter in `run_integration_tests.sh` silently excludes models lacking `requirements.txt`

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | A user runs `bash run_integration_tests.sh --library baseline` and one of the eligible models is missing `requirements.txt` (or the file is unreadable); the model is excluded from the run with no warning |
| **Source** | code-review (2026-04-11) — discovered during C-26 fix in Sprint 2 |
| **Status** | Open |
| **Notes** | `run_integration_tests.sh:128-137` uses `if [ -f "$req_file" ] && grep -q "views-${FILTER_LIBRARY}" "$req_file"`. A missing or unreadable `requirements.txt` causes silent exclusion — the same class of bug C-26 had in the `--level` filter, but in the `--library` filter. C-26's Sprint 2 fix added the `CLASSIFICATION_ERRORS` fail-fast pattern (lines 109-153) for level classification only; the library filter was left untouched because it does not crash and the legitimate "model declares no matching library" case must remain a silent skip. The remaining gap: a model that lacks `requirements.txt` entirely cannot be distinguished from one that declares a different library. **Recommended fix:** when `requirements.txt` does not exist for a model in the candidate set, emit a `WARNING: cannot classify <model> by library: missing requirements.txt` to stderr and exclude it explicitly (don't fail fast — this is milder than C-26 because it doesn't indicate a broken file). After C-08 (requirements coherence test) and C-27 (rude_boy backfill), this gap is mostly future-protection — it would re-emerge if a new model is added without `requirements.txt` and `--library` filtering is used before C-08 catches the omission. See also C-26 (same pattern, resolved 2026-04-11), C-08 (requirements coherence — mitigated), C-27 (rude_boy `requirements.txt` — resolved). **Cluster G** (added 2026-10-26, #610) — the `--library` filter excludes a model for lacking the very file the filter reads, with no warning, and the run reports on the remainder as though it were the roster. |

---

### C-35 — No CI gate for CIC ↔ code synchronization

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A PR modifies behavior of a CIC-governed class (anything in `docs/CICs/*.md`) — new guarantees, new failure modes, new inputs, new exit codes, new outputs — without updating the corresponding CIC file in the same PR, and merges without the drift being flagged |
| **Source** | review-diff (2026-04-11) — discovered during PR review of `fix/hydranet_loss_hp` |
| **Status** | Resolved |
| **Notes** | ADR-006 requires CIC updates to follow behavioral changes ("Changes to intent must update this contract," quoted at the bottom of every CIC). The repo enforces this via social review, not automation: nothing in `.github/workflows/` or `tests/` verifies that CIC-governed files have not drifted from their CIC. **Concrete evidence (this PR):** three commits to `run_integration_tests.sh` (`97aeb38` added DEPRECATED skip + exit code 130; `cd668ea` unrelated but didn't touch the CIC; `1ea564c` added `--foreground` changing signal semantics) shipped before review-diff flagged that `docs/CICs/IntegrationTestRunner.md` sections 3 (guarantees), 6 (failure modes table), and 7 (boundaries) still described the pre-change behavior. Each commit passed all pytest checks and was individually reviewed, yet the CIC drift went uncaught for three iterations. The test suite (3312 passing) has zero cross-references between CIC content and code behavior. **Why this matters beyond this PR:** CICs are load-bearing documentation for onboarding, incident response, and upstream contract negotiation (e.g., the C-31 pandas incident relied on CICs to understand the boundary between views-models and views-stepshifter). Stale CICs give readers a confidently wrong mental model. The bigger the drift, the worse the misdirection. **Recommended fix (not in scope for this concern):** a CI check that, for every file under `docs/CICs/`, enforces "if the target code file(s) changed in this PR, the CIC must also have changed in this PR." The challenge is mapping CIC → target files; the CIC filename already names the class, and a one-line frontmatter field (e.g., `target: run_integration_tests.sh`) plus a 30-line `.github/workflows/cic_sync_check.yml` would suffice. Related: C-15 (zero CIC failure mode test coverage — specifically about testing declared failure modes), C-16 (zero direct unit tests on CIC classes — specifically about behavior coverage), C-07 (scaffold builder testing gap). This concern is distinct: it's about documentation drift, not test coverage. |

---

### C-36 — `create_catalogs.py` uses fixed module names in `importlib` loading, risking stale module cache

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A Python runtime or future code change registers importlib-loaded modules in `sys.modules`; subsequent `extract_models()` calls return config data from the wrong model |
| **Source** | review-diff (2026-04-20) |
| **Status** | Resolved |
| **Location** | `create_catalogs.py:48,57` |
| **Notes** | `spec_from_file_location("config_meta", config_meta)` reused the literal name `"config_meta"` for every model's config file. Fixed (2026-04-21): module names now include the model directory name (`f"config_meta_{model_dir_name}"`), matching the `conftest.py:load_config_module` pattern. See also C-17, C-19. |

---

### C-37 — bright_starship `config_partitions.py` uses `_current_month_id()` instead of `ViewsMonth`, creating test blind spot

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | The `ViewsMonth` epoch or convention diverges from `(year - 1980) * 12 + month`, or a developer relies on `test_config_partitions.py` passing as proof that bright_starship's forecasting offset is correct |
| **Source** | review-diff (2026-04-20) |
| **Status** | Mitigated |
| **Location** | `models/bright_starship/configs/config_partitions.py:17-20,35` |
| **Notes** | bright_starship reimplements `ViewsMonth.now().id` as `_current_month_id()` to avoid `ingester3` dependency. The test regex finds zero matches, so the offset check vacuously passes. **Mitigated (2026-04-21):** added `# PARTITION_OVERRIDE:` comment so the test framework explicitly skips with a warning rather than silently passing. **2026-05-20 (fix):** Removed `_current_month_id()` from all 4 synthetic entries (vertical_dream, horizontal_dream, diagonal_dream, synthetic_chorus) by replacing dynamic forecasting ranges with fixed boundaries — train (121, 540), test (541, 541 + steps). Synthetic data has no external data availability constraint so fixed ranges are sufficient. These files no longer carry the epoch-divergence risk. Residual risk applies only to bright_starship, heavy_strider, heavy_freighter, light_strider, and shining_codex (all carry `# PARTITION_OVERRIDE:` comments). **2026-05-26 (ensemble parity dimension):** bold_comet, blazing_meteor, and stellar_horizon also use `_current_month_id()`. golden_hour (viewser ensemble) uses `ViewsMonth`. When comparing golden_hour ↔ stellar_horizon forecasting parity, the two implementations may disagree by ±1 month at month boundaries, silently shifting the forecasting train/test partition and invalidating the comparison. `test_datafactory_parity.py` only checks calibration/validation boundaries (static, identical) — it does not catch forecasting divergence. Forecasting parity comparisons in the runbook (Phase 7) must account for this. See also C-01, D-01. |

---

### C-38 — `datafactory_query` not installed in any environment that can run bright_starship

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A developer runs `python main.py -r calibration` in `views-hydranet-env` (or any env with `views_hydranet` + `views_pipeline_core`) without `datafactory_query` installed, and `calibration_viewser_df.parquet` is not cached |
| **Source** | falsify (2026-04-21) |
| **Status** | Open |
| **Location** | `models/bright_starship/main.py:33` (`from configs.config_queryset import fetch_data`), `models/bright_starship/configs/config_queryset.py:115` (`from datafactory_query import load_dataset`), `models/shining_codex/main.py:27` (same pattern), `models/shining_codex/configs/config_queryset.py:90` (same pattern) |
| **Notes** | **Falsification audit F-1/F-2 chain.** `views-datafactory` (which provides `datafactory_query`) is declared in `requirements.txt` but not installed in `views-hydranet-env` — the only conda environment that has both `views_hydranet` and `views_pipeline_core`. When `_ensure_data()` encounters a cache miss, it imports `datafactory_query` at line 96 and crashes with `ModuleNotFoundError`. Two of three run_types (`validation`, `forecasting`) have cached parquets from a prior session, masking the missing dependency. `calibration` has no cache — the standard first run (`-r calibration -t -e`) fails immediately. The local `envs/views-hydranet` directory expected by `run.sh` also does not exist; `run.sh` would create it and install deps from `requirements.txt` (which includes the git+https datafactory dep), but that's a ~10 min bootstrap, not "ready to run." **Fix:** `conda run -n views-hydranet-env pip install "views-datafactory>=1.9.0"` (on PyPI since 2026-07-27). See also C-06 (config_queryset external deps — accepted for viewser; this is the datafactory equivalent), C-37 (bright_starship partition deviation), C-40 (generate() contract mismatch). **Cross-repo (IDs below belong to the *views-pipeline-core* register, NOT this one — the same numbers exist here with unrelated content):** `vpc C-51` (`get_data()` hardcodes viewser), `vpc C-52` (drift detection loss), `vpc C-53` (`use_saved` overload). **2026-06-12:** the bright_starship half is fixed on this workstation — `views-hydranet-env` now has datafactory_query and the readiness probe passes locally. Still open for shining_codex (`views-r2darts2` env unprovisioned; its probe skips) and for any fresh machine — keep Open until the env story (run.sh bootstrap or release-pinned install) is settled. **Tier recalibrated 2 → 3 during review-rr (2026-07-31):** the failure is a loud `ModuleNotFoundError` at first run — provisioning friction, not structural fragility with a silent consequence. Demoted alongside C-42, C-50 and C-73 so the Tier-2 band means "silent or stakeholder-visible", not "annoying on a fresh clone". Member of **Cluster C** (cross-repo dependencies have no released contract). **Root cause registered 2026-08-02:** this is a specific instance of **C-116** — 131 `requirements.txt` resolve into 11 shared environments, so a package a model needs can be absent because a co-tenant's run shaped the environment. Fix the class from C-116; this entry stays as the concrete instance that surfaced it. |

---

### C-39 — All 70 `run.sh` scripts use `#!/bin/zsh` — will fail on Linux servers and CI

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Any `run.sh` is executed on a Linux server, Docker container, or CI runner where zsh is not installed (i.e., most deployment targets) |
| **Source** | review-diff (2026-04-21); **regression measured 2026-08-02** while scoping #310 |
| **Status** | Resolved (2026-08-02, second time) — see the regression note |
| **Location** | `models/*/run.sh`, `ensembles/*/run.sh`, `apis/*/run.sh`, `extractors/*/run.sh`, `postprocessors/*/run.sh`, `models/execute_all.sh` (82 scripts total) |
| **Notes** | **REGRESSED between 2026-05-04 and 2026-06-28, while marked Resolved; re-fixed 2026-08-02 (#310).** 24 `run.sh` carrying `#!/bin/zsh` were found. Every one was created *after* the April fix — 2026-05-04 (`first_love`, `bad_romance`, `smol_cat`, others), 2026-05-19 (`fake_model`), 2026-06-28 (the 12 r2darts models). They are not stragglers a sweep missed. **Cause: the fix was applied to the output, never to the generator.** `run.sh` is emitted by `views_pipeline_core/templates/model/template_run_sh.py` in **views-pipeline-core**, which still emits `#!/bin/zsh` and was last modified 2026-04-03 — eighteen days *before* the April fix landed here, by the same author. Every model scaffolded since has been born with the defect. **Impact, measured:** `models/execute_all.sh:10` invokes `"$script"` directly and every ensemble README documents `./run.sh`, so on Linux 11 of the 24 (zsh *and* executable) fail with `bad interpreter`. One is `ensembles/first_love`, which `monthly_run.sh` runs in production — undetected because `monthly_run.sh` calls `bash run.sh`, which ignores the shebang. **Exit:** `tests/test_run_sh_portability.py` now fails on any tracked `.sh` declaring zsh, so the next generator-sourced regression is caught on the day it lands rather than eight weeks later. The generator itself is **not fixed here** — that is another repository (**views-pipeline-core#384**), so this entry stays exposed to re-regression until that lands; the test is what makes that visible. **The executable bit, the second half of the same failure (fixed 2026-08-02 on maintainer decision):** 18 tracked `run.sh` were committed non-executable (mode 100644), breaking the same two entry points with `Permission denied`; 13 overlapped the zsh set, so those went from one Linux failure straight to another. Now mode 100755 and pinned by `test_every_run_sh_is_executable`. One deliberate exception, also pinned: `tools/credentials/platform_env.sh` is `source`d and never executed (ADR-018), so an executable bit there would advertise an entry point it does not have. **The pattern:** third entry this quarter marked Resolved while a generator or a scope kept producing the defect (C-60 tools layout, C-112 shell-vs-exported scope). Cross-refs: **C-60**, **C-112**, **C-113**. Member of **Cluster A** (declared-but-unenforced). |
| **Original notes (2026-04-21)** | **Resolved (2026-04-21).** All 79 `#!/bin/zsh` shebangs changed to `#!/usr/bin/env bash`. `models/execute_all.sh` line 10 changed from `zsh "$script"` to `"$script"` (delegates to shebang). 35 missing trailing newlines and 23 missing executable permissions also fixed. `scripts/audit_shell_health.sh` added to verify: 82 scripts, 490 checks, CLEAN verdict. No zsh-specific syntax was found in any script — all were plain POSIX/bash. |

---

### C-40 — `generate()` return type contract mismatch — dict vs Queryset, no validation

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A new model migrates to views-datafactory and its `config_queryset.generate()` returns a dict descriptor; `views-pipeline-core` calls `.publish()` on it and crashes |
| **Source** | expert-code-review (2026-04-21) |
| **Status** | Open |
| **Location** | `models/bright_starship/configs/config_queryset.py` (returns dict), `models/shining_codex/configs/config_queryset.py` (returns dict), `views-pipeline-core/views_pipeline_core/data/model_path.py:691-692` (`get_queryset()` returns raw `generate()` output with no type checking) |
| **Notes** | Standard viewser models return a `Queryset` object from `generate()`. bright_starship and shining_codex (datafactory models) return a plain dict with `"source": "views-datafactory"`, `"zarr_url"`, `"features"` keys. `get_queryset()` in views-pipeline-core performs no type checking — it calls `generate()` and returns whatever it gets. Downstream, `_fetch_data_from_viewser()` calls `.publish()` on the result, crashing with `AttributeError: 'dict' object has no attribute 'publish'`. The contract between views-models (config producer) and views-pipeline-core (config consumer) is entirely implicit. **Phase 1 workaround:** `args.saved = True` in bright_starship's `main.py` routes around the viewser path. **Phase 2 fix (views-pipeline-core):** type dispatch in `get_data()` based on descriptor type + `generate()` return type validation in `get_queryset()`. **Cross-repo (IDs below belong to the *views-pipeline-core* register, NOT this one):** `vpc C-51` (root cause — `get_data()` hardcodes viewser), `vpc C-42` (missing ViewsDataLoader CIC). See also C-06 (config_queryset external deps), C-38 (datafactory_query not installed). |

---

### C-41 — shining_codex has no readiness tests

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A developer clones the repo and runs `python main.py -r calibration` for shining_codex without the `views-r2darts2` environment and `datafactory_query` installed |
| **Source** | tech-debt-cleanup (2026-04-21) |
| **Status** | Resolved (2026-06-12) |
| **Location** | `models/shining_codex/` (no `tests/` directory or test files) |
| **Notes** | bright_starship has readiness tests (`test_bright_starship_readiness.py`) that verify environment prerequisites (conda env, `datafactory_query`, `DartsForecastingModelManager` import) and config structural validity. shining_codex, cloned from bright_starship, has no equivalent tests. Without readiness tests, failures will surface only at runtime with opaque error messages (e.g., `ModuleNotFoundError` for `datafactory_query` or `views_r2darts2`). See C-38 (datafactory_query not installed), C-03 (integration tests manual-only). **2026-06-12: Resolved** (issue #122, with C-75): `test_bright_starship_readiness.py` is parametrized over both datafactory models — shining_codex gets the same env pre-flight probe (skips while `views-r2darts2` is unprovisioned, which is truthful) and the same static dependency-contract checks (requirements / queryset import / generate()). Single parametrized file avoids the copy-paste drift this entry complained about. |

---

### C-42 — Synthetic models depend on unreleased `views-pipeline-core` branch

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | The `feature/hydranet_ensamble_africa_me` branch of views-pipeline-core changes its synthetic data API (pattern names, queryset descriptor keys, or `DataFrameEnsembleManager`/`PredictionFrameEnsembleManager` constructor) before merge, breaking synthetic models and ensembles |
| **Source** | pr-review (2026-05-20) |
| **Status** | Resolved (2026-09-15) |
| **Location** | `models/vertical_dream/configs/config_queryset.py`, `models/horizontal_dream/configs/config_queryset.py`, `models/diagonal_dream/configs/config_queryset.py`, `ensembles/synthetic_chorus/main.py`, `models/lucid_dream/configs/config_queryset.py`, `models/vivid_dream/configs/config_queryset.py`, `models/waking_dream/configs/config_queryset.py`, `ensembles/synthetic_chant/main.py` |
| **Notes** | PR #56 adds `vertical_dream`, `horizontal_dream`, `diagonal_dream`, and `synthetic_chorus` — all four depend on the `"source": "synthetic"` queryset descriptor and `DataFrameEnsembleManager`, which exist only on the `feature/hydranet_ensamble_africa_me` branch of `views-pipeline-core`. If that branch renames pattern values (e.g., `"vertical_stripe"` → `"v_stripe"`), changes required descriptor keys, or alters the `EnsembleManager` import path, the synthetic models will fail at data-load time with no structural test catching the mismatch — `test_model_structure.py` validates directory layout but not queryset descriptor validity against pipeline-core. This is the same class of cross-repo coupling as C-31 and C-38 but with a sharper trigger: the dependency is on an unreleased, in-flux branch rather than a released package. Risk resolves naturally once the pipeline-core branch merges and the API stabilizes. **2026-05-24 (PR #58):** Three additional PredictionFrame synthetic models (`lucid_dream`, `vivid_dream`, `waking_dream`) and one ensemble (`synthetic_chant`) added. These extend the dependency surface to `PredictionFrameEnsembleManager`, `ConflictologyModel`, and `MixtureBaseline` distributional outputs. All run successfully against `views-pipeline-core v2.3.0` — if that version is released, this risk may be resolved. **2026-05-26 (confirmed):** `envs/views_ensemble` created by ensemble `run.sh` installs `views-pipeline-core` from PyPI, which lacks `PredictionFrameEnsembleManager`. `synthetic_chant` ensemble failed with `ImportError: cannot import name 'PredictionFrameEnsembleManager'` until local editable install replaced the PyPI version. This confirms the trigger: any fresh clone or CI environment that creates `views_ensemble` from `requirements.txt` will fail for PredictionFrame ensembles. See also C-31 (upstream API breaks), C-38 (datafactory_query not installed), C-40 (generate() return type contract mismatch), C-50 (views-baseline version spec mismatch — same class of fresh-clone failure). **Tier recalibrated 2 → 3 during review-rr (2026-07-31):** the confirmed failure is a loud `ImportError` at ensemble start on any fresh env — provisioning friction, not silent fragility. Member of **Cluster C** (cross-repo dependencies have no released contract). **Resolved (2026-09-15):** the branch merged and released. `PredictionFrameEnsembleManager` ships in every published views-pipeline-core 3.x — verified by import from 3.2.0 — and CI pins `views_pipeline_core==3.0.1` (`run_tests.yml:30`). The 13 ensembles declare `>=3.0.0,<4.0.0` (#372). |
---

### C-43 — Ensemble ground truth is order-dependent on `config_meta.models` list

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | A developer reorders the `models` list in `ensembles/synthetic_chorus/configs/config_meta.py` |
| **Source** | falsify audit (2026-05-20) |
| **Status** | Open |
| **Location** | `ensembles/synthetic_chorus/configs/config_meta.py:4` |
| **Notes** | The ensemble evaluation loads prediction files from constituent models in list order. The actual `synth_target` values (ground truth) come from the first model's predictions — currently `vertical_dream`. The analytically derived expected MSE (4.34444) depends on this ordering. Reordering the list silently changes which model supplies the ground truth, producing a different MSE with no error signal. Mitigated by `tests/test_falsification_synthetic.py::test_synthetic_chorus_first_model_is_vertical_dream` which asserts vertical_dream is first, and by the README which documents the order-dependency. This is a test-internal concern with no production impact — synthetic models are not deployed. |

---

### C-44 — Quality-blind ensemble aggregation (concat *and* mean) degrades the ensemble below its best constituents

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Building a `concat` **or** `mean` ensemble whose constituents have heterogeneous quality on a target (one constituent materially worse than others), with `aggregation` set as a bare config string and no constituent-quality gate |
| **Source** | golden_hour calibration run (2026-05-25); broadened by repo-assimilation + chunky_bunny (2026-06-13) |
| **Status** | Open |
| **Location** | `ensembles/*/configs/config_meta.py` (`aggregation`); `views-pipeline-core` ensemble aggregation paths (PredictionFrameEnsembleManager concat; mean) |
| **Notes** | Observed 53% CRPS degradation on `lr_sb_best` vs best individual model (golden_hour: 0.233 vs purple_alien: 0.152). blue_stranger (0.223) contributed 64 poor-quality samples that diluted the 128 better samples from purple_alien and violet_visitor. Concat treats all posterior samples equally — no mechanism to down-weight poor contributors. For future ensembles, consider weighted aggregation or model selection for targets where constituent quality varies significantly. Models were uncalibrated so this finding may not hold after hyperparameter optimization. **2026-06-13 (repo-assimilation R7 — merged here): the same quality-blindness affects `aggregation: "mean"`.** chunky_bunny (equal-weight mean of 23 constituents) scored MSLE **0.590**, worse than 14 of its own constituents and Pareto-dominated by smol_cat alone (0.503 MSLE / 0.872 MCR vs the ensemble's 0.590 / 0.584): the mean blends timid stepshifters (MCR 0.2–0.3) with honest DL/Hurdle models and lands in a mediocre middle. `aggregation` is a bare string in `config_meta.py` with no quality gate, for either method. See also C-13 (no prediction quality validation before aggregation), C-86 (constituent feature incoherence), [[project-mcr-timid-prophet]]. |

---

### C-45 — Ensemble `-t` flag causes full retraining cascade when models are pre-trained

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Running any PredictionFrameEnsembleManager or DataFrameEnsembleManager with `-t` when constituent models already have trained artifacts in their `artifacts/` directories |
| **Source** | golden_hour calibration run (2026-05-25) |
| **Status** | Open |
| **Location** | `views-pipeline-core` EnsembleManager train path (invokes constituent `run.sh` subprocesses) |
| **Notes** | Running `python main.py -r calibration -t -e` on a pre-trained ensemble causes: (1) retrain all constituent models via run.sh subprocess (~2h), (2) create new model artifacts with new timestamps, (3) discover no predictions exist for those new timestamps, (4) re-evaluate all constituent models via run.sh subprocess (~3h), (5) finally perform the actual aggregation (~30 min). This wasted ~6 hours on golden_hour. The correct command when models are already trained: `python main.py -r calibration -e --saved`. The `-t` flag on ensembles should either warn when artifacts already exist, or detect and reuse existing timestamps rather than creating new ones. **Tier recalibrated 2 → 3 during review-rr (2026-07-31):** the consequence is ~6 hours of wasted compute — fully observable, recoverable, and it produces no wrong output. Costly, not fragile. See also C-14 (artifacts have no run identity — the timestamp churn this entry describes is the same missing identity; absorbed C-22). |

---

### C-46 — Classification targets not evaluable at PredictionFrame ensemble level

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Adding `classification_targets` to any PredictionFrame ensemble's `config_meta.py` |
| **Source** | golden_hour design review (2026-05-24) |
| **Status** | Open |
| **Location** | `views-pipeline-core` `PredictionFrameEnsembleManager.prepare_actuals_df` (identity lambda), `ensembles/golden_hour/configs/config_meta.py` (regression-only by design) |
| **Notes** | `PredictionFrameEnsembleManager.prepare_actuals_df` is a no-op identity lambda. Classification targets (`by_sb_best`, `by_ns_best`, `by_os_best`) are derived signals not present in raw viewser data. Individual HydraNet models derive them via `DataFetcher.apply_blueprint()`, but the ensemble doesn't inherit that derivation logic. Including `classification_targets` in ensemble `config_meta` causes `KeyError` when `EvaluationStage._load_actuals()` looks for the derived columns in raw actuals. Workaround: exclude `classification_targets` from ensemble config; evaluate classification at individual model level only. golden_hour correctly implements this workaround. Fix would require `PredictionFrameEnsembleManager` to implement target derivation or delegate to constituent model blueprints. See also C-15 (CIC failure mode coverage — ensemble aggregation failure modes listed as remaining gap). |

---

### C-47 — Track A/B dual output produces redundant predictions with contradictory documentation

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A developer sets `skip_predictions_delivery` back to `False` to re-enable Track B parquets without verifying the PyArrow memory fix is in place |
| **Source** | golden_hour investigation (2026-05-25) |
| **Status** | Mitigated |
| **Location** | `views-pipeline-core` config (`skip_predictions_delivery` flag), `models/*/data/generated/` (both `.npy` and `.parquet` outputs coexist) |
| **Notes** | HydraNet models produce both Track A (`.npy` PredictionFrame, 64 posterior samples) and Track B (`.parquet` DataFrame delivery, point predictions) simultaneously. **Mitigated (2026-05-26):** All 19 PredictionFrame models now have `skip_predictions_delivery: True`, suppressing Track B parquet generation. The contradictory `False, #True,` comment pattern has been removed. `test_track_parity.py` (40 tests) verified Track A and Track B produce identical values before Track B was disabled. `CoreConfigSniffer` (views-pipeline-core PR #87) now enforces the key as mandatory — models without it crash at config validation. Residual risk: if Track B is re-enabled without the PyArrow memory fix, the 5.5M Python float object allocation (~4.8–6.4 GB peak) will recur. See also C-40 (generate() return type contract mismatch). |

---

### C-48 — Viewser vs datafactory variable variant mismatch confounds parity comparison

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | CRPS or forecast parity comparison between golden_hour (viewser) and stellar_horizon (datafactory) shows divergence; root cause is data input differences, not pipeline differences |
| **Source** | config diff investigation (2026-05-26) |
| **Status** | Resolved |
| **Location** | `models/purple_alien/configs/config_queryset.py` (`ged_sb_best_sum_nokgi`), `models/bright_starship/configs/config_queryset.py` (`ged_sb_best`), same pattern for `ged_ns_best` and `ged_os_best` |
| **Notes** | The viewser trio (purple_alien, blue_stranger, violet_visitor) trains on `ged_*_best_sum_nokgi` — the summed, no-known-geographical-imprecision variant of UCDP fatality counts. The datafactory trio (bright_starship, bold_comet, blazing_meteor) trains on `ged_*_best` — the base variant. Despite different variable names, both deliver functionally identical values. **Resolved (2026-05-26):** Direct cell-by-cell comparison of cached training parquets (4,876,920 rows × 6 columns) showed 99.99% exact match for all three target variables: lr_sb_best (614 differing rows of 4.9M), lr_ns_best (138), lr_os_best (182). Correlations all >0.999. The `_sum_nokgi` suffix does not indicate a different aggregation — both sources deliver the same fatality sums per PRIO-GRID cell-month. The ~600 differing rows have small absolute differences reflecting timing differences in UCDP data ingestion. This concern is fully disproven as a source of prediction divergence. See `reports/parity_investigation_20260526.md` for full analysis. See also C-02 (queryset validation), C-40 (generate() contract mismatch). **2026-06-24 (un_fao delivery consumer, #94):** The same variant pair now feeds the **un_fao postprocessor**, which switched from viewser `*_sum_nokgi` to datafactory `*_best` (commit 2518335) and *delivers per-cell actuals to FAO* — a stricter consumer than model training. A row-level oracle over `africa_me_legacy`, months 480–485 (`tests/test_un_fao_datafactory_equivalence.py`) confirms C-48's finding at the delivery level: coverage identical (78,660 cells, none added/dropped), values 99.97% identical (~21 cells differ across the 3 targets, net +15/+14/+39 fatalities) — consistent with ingestion-timing skew, not a structural aggregation change. The test enforces this as a **bounded-divergence guard** (cell-fraction ≤1%, net ≤10% of total per target) that tolerates the skew but fails loud on a real aggregation/region change; a strict bit-equality test is kept `xfail` to document the non-identity. Net-positive drift (datafactory slightly higher) is within noise for this window but worth re-checking before #127 goes global to `land_gaul` (5× more cells → 5× the absolute skew in delivered numbers). |

---

### C-49 — Feature set divergence between viewser and datafactory model configs

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | Parity comparison between golden_hour and stellar_horizon produces unexplained spatial or regional bias differences |
| **Source** | config diff investigation (2026-05-26) |
| **Status** | Partially Resolved |
| **Location** | `models/purple_alien/configs/config_queryset.py` (lines 22-23: `col`, `row` columns), `models/bright_starship/configs/config_queryset.py` (no spatial features) |
| **Notes** | Originally three concerns. **Partially resolved (2026-05-26):** **(1) Spatial features: DISPROVEN.** Raw data comparison confirmed `col` and `row` are 100% identical between viewser and datafactory parquets. Both data loading paths provide them. **(2) Country encoding: CONFIRMED but metadata-only.** viewser uses VIEWS-internal `country_id` (e.g., 192); datafactory uses FAO `gaul0_code` (e.g., 159, or -1 for unassigned). 0% cell-level match. However, `c_id` is in `identity_cols`, NOT in `features` — HydraNet uses only 3 input channels (lr_sb_best, lr_ns_best, lr_os_best). Unless curriculum sampling or stratified evaluation uses `c_id` values downstream, this is a metadata-only divergence with no model impact. Downgraded from Tier 2 to Tier 4. **(3) NA handling:** Not yet investigated. See `reports/parity_investigation_20260526.md` for full analysis. See also C-48 (resolved — variable variant not a divergence source), C-02 (queryset correctness). |

---

### C-50 — `views-baseline` not published to PyPI; `requirements.txt` version spec unresolvable on fresh clone

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A developer clones the repo on a new machine and runs any baseline model's `run.sh`, which creates `envs/views-baseline` and fails at `pip install -r requirements.txt` because `views-baseline>=1.0.0,<2.0.0` has no matching distribution on PyPI |
| **Source** | synthetic ensemble run (2026-05-26) |
| **Status** | Resolved (2026-09-15) |
| **Location** | `models/lucid_dream/requirements.txt`, `models/vivid_dream/requirements.txt`, `models/waking_dream/requirements.txt`, `models/vertical_dream/requirements.txt`, `models/horizontal_dream/requirements.txt`, `models/diagonal_dream/requirements.txt`, `models/red_ranger/requirements.txt`, `models/green_ranger/requirements.txt`, `models/blue_ranger/requirements.txt`, `models/black_ranger/requirements.txt`, `models/pink_ranger/requirements.txt`, `models/yellow_ranger/requirements.txt`, `models/white_ranger/requirements.txt`, `models/light_strider/requirements.txt`, `models/heavy_strider/requirements.txt`, `models/average_cmbaseline/requirements.txt`, `models/average_pgmbaseline/requirements.txt`, `models/zero_cmbaseline/requirements.txt`, `models/zero_pgmbaseline/requirements.txt`, `models/locf_cmbaseline/requirements.txt`, `models/locf_pgmbaseline/requirements.txt` (21 models total) |
| **Notes** | All 21 baseline models declare `views-baseline>=1.0.0,<2.0.0` in `requirements.txt`. The `views-baseline` package is not published to PyPI at all — it is only available as a local editable install from `~/Documents/scripts/views_platform/views-baseline` at version `0.1.0`. On existing developer machines with the pre-existing `envs/views-baseline` env, the pip dry-run check succeeds because the package is already installed, and `run.sh` proceeds normally. On a fresh clone (new machine, CI, new contributor), `run.sh` creates the conda env, `pip install` fails with `No matching distribution found for views-baseline`, and the model crashes with `ModuleNotFoundError: No module named 'views_baseline'`. **Observed (2026-05-26):** All 6 synthetic model runs showed `ERROR: No matching distribution found for views-baseline<2.0.0,>=1.0.0` but succeeded because the env already had the local install. **Fix options:** (1) publish `views-baseline` to PyPI at version `>=1.0.0`, (2) change `requirements.txt` to use a git+https URL (matching the `views-datafactory` pattern in HydraNet models), (3) update `run.sh` to install from local path if available (but run.sh must not be modified — see feedback constraint). See also C-38 (same class: `datafactory_query` not installed), C-42 (same class: `views-pipeline-core` from PyPI lacks features), C-08 (requirements coherence). **Tier recalibrated 2 → 3 during review-rr (2026-07-31):** `No matching distribution found` is a loud, immediate, self-describing failure — provisioning friction, not silent fragility. Member of **Cluster C** (cross-repo dependencies have no released contract); the cluster fix is a release/pin discipline, of which publishing `views-baseline` is one instance. **Resolved (2026-09-15):** views-baseline has been on PyPI since 1.0.0 (2026-07-28); 1.0.2 (2026-09-09) is the first version that runs — 1.0.0 and 1.0.1 read the `targets` key pipeline-core retired (#445). All 37 baseline models pin `>=1.0.2,<2.0.0` (#460), verified installable in an empty venv, and `runtime_smoke.yml` installs it from PyPI. Fix option (1) is what happened. |
---

### C-51 — Datafactory trio missing `sampling_strategy` — ADR-049 required field, runtime crash

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A developer runs `bash models/bold_comet/run.sh -r calibration` (or bright_starship, blazing_meteor) and views-hydranet rejects the config with `'sampling_strategy' is required (ADR-049)` |
| **Source** | review (PR #59, 2026-05-31) |
| **Status** | Resolved |
| **Location** | `models/bright_starship/configs/config_hyperparameters.py`, `models/bold_comet/configs/config_hyperparameters.py`, `models/blazing_meteor/configs/config_hyperparameters.py`, `models/heavy_freighter/configs/config_hyperparameters.py` |
| **Notes** | The viewser trio (purple_alien, blue_stranger, violet_visitor) received `sampling_strategy` in this PR cycle (threshold/boltzmann/sigmoid respectively). The datafactory trio and heavy_freighter were not updated — bold_comet and blazing_meteor were cloned from bright_starship, which also lacked the field. views-hydranet's curriculum learner validates the key at config load time and raises `KeyError` on absence. All four models would fail immediately on any training run. The parity test (`test_datafactory_parity.py::TestDatafactoryTrioConfigParity::test_identical_shared_hyperparameters`) does not catch this because it strips loss keys and compares models pairwise — since all three are equally missing the field, they match each other. **Resolved (2026-06-01):** Added `'sampling_strategy': 'threshold'` to all four affected models (3 datafactory + heavy_freighter). Added `test_hydranet_has_sampling_strategy` to `test_config_completeness.py` to catch this class of omission for all HydraNet models (scoped via `meta_config["algorithm"] == "HydraNet"`) — this test is what caught heavy_freighter. See also C-05 (incomplete HP validation — covers stepshifter/baseline, not HydraNet), C-38 (datafactory_query not installed — same models, different dependency class), C-42 (unreleased pipeline-core branch — different: import availability, not config completeness). |

---

### C-52 — 12 PF models missing config keys required for PFE ensemble participation

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A developer adds any of the 12 affected models as a constituent of a PredictionFrameEnsembleManager ensemble — the ensemble will crash or produce wrong sample counts because constituent configs lack `n_posterior_samples` and/or `regression_targets` |
| **Source** | test_pfe_production_readiness.py (TDD green tests, 2026-06-01) |
| **Status** | Resolved |
| **Location** | `models/{black_ranger,blue_ranger,green_ranger,lucid_dream,pink_ranger,red_ranger,vivid_dream,waking_dream,yellow_ranger}/configs/config_hyperparameters.py` (missing both `n_posterior_samples` and `regression_targets`), `models/{heavy_strider,light_strider,white_ranger}/configs/config_hyperparameters.py` (missing `n_posterior_samples` only) |
| **Notes** | All 21 models declare `prediction_format: "prediction_frame"` in `config_meta.py`, meaning they produce PredictionFrame outputs. But 12 of them lack `n_posterior_samples` (needed by PFE to verify aggregated sample counts) and 9 of those also lack `regression_targets` (needed to know which target directories to validate). The 9 models with fully compliant configs (purple_alien, blue_stranger, violet_visitor, bright_starship, bold_comet, blazing_meteor, heavy_freighter, pink_pirate, heavy_strider partially) are the only ones eligible for PFE ensembles today. This blocks the PFE production roadmap: Steps 2-5 require running constituent models through PFE, and any model without these keys cannot participate. The ranger models (7 of 12) use an older config convention with `n_samples` instead of `n_posterior_samples` and no explicit `regression_targets` — they predate the HydraNet multi-target architecture. The dream models (lucid_dream, vivid_dream, waking_dream) are synthetic test models that also predate the convention. **Resolved (2026-06-02):** Added `n_posterior_samples` and `regression_targets` to all 12 affected `config_hyperparameters.py` files. Values derived from each model's `config_meta.py` (regression_targets) and existing `n_samples` (n_posterior_samples). xfail markers removed from `test_pfe_production_readiness.py` — all 21 PF models now pass config-level readiness tests unconditionally. See #70. |

---

### C-53 — Config value regression during cross-branch merges

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A developer merges `development` into a feature branch (or vice versa) when both branches have modified the same model `config_hyperparameters.py` with different values for the same key — git auto-resolves by picking one side, silently dropping the other's intentional change |
| **Source** | tech-debt-cleanup (2026-06-02) |
| **Status** | Open |
| **Location** | `models/blue_stranger/configs/config_hyperparameters.py`, `models/violet_visitor/configs/config_hyperparameters.py` (observed); any model config modified on both branches (systemic) |
| **Notes** | Observed during merge of `development` into `feature/golden_hour_ensemble`: blue_stranger and violet_visitor had `skip_predictions_delivery` changed to `True` on the feature branch (intentional), while development still had `False` (pre-existing). Git auto-merged without conflict markers, silently regressing the value to `False`. Also introduced a stray `prediction_format` key in hyperparameters (belongs only in config_meta). Caught during tech-debt-cleanup verification; would have caused Track B parquet generation and potential OOM in ensemble runs. **Mitigated (2026-06-02):** Fixed in this session. No automated guard exists — mitigation is manual post-merge review of config diffs. See also C-01 (73 duplicated config files amplify this risk), C-52 (same files, different keys). |

---

### C-54 — Experimental model (heavy_freighter) in production model directory without marker

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | **(a) Original —** a developer adds heavy_freighter to a production ensemble's `config_meta.models` list without realizing it uses a global grid (360×720 vs regional 180×180), producing incompatible spatial dimensions. **(b) Broadened (2026-07-20) —** a cross-cutting sweep (partition bump, config-key migration, catalog regeneration, an all-models parametrized test) runs over `models/` and silently includes the ~30 experimental/placeholder directories, because no marker distinguishes them |
| **Source** | tech-debt-cleanup (2026-06-02) |
| **Status** | Open |
| **Location** | `models/heavy_freighter/configs/config_hyperparameters.py` (`height: 360`, `width: 720` — global grid vs regional 180×180) |
| **Notes** | heavy_freighter uses global grid coverage (360×720) vs the regional Africa-ME grid (180×180) used by all ensemble-eligible models. Its training params (tobit, 200 lessons, 16 samples, scheduled sampling) now match the production models — only the grid differs. It is correctly excluded from golden_hour and stellar_horizon ensembles. The risk is that no directory convention, marker file, or test distinguishes global-grid models from regional models — the only signal is reading the config. Low severity because incompatible spatial dimensions would cause a shape mismatch error at ensemble aggregation time. **Broadened (repo-assimilation 2026-07-20):** heavy_freighter is one instance of a wider gap — `models/` now holds **120 directories**, a large fraction experimental/placeholder (8 `temporary_*`, the `*_dream`/synthetic families, `*_ranger` families, `*_dwarf` experiment set) with **no lifecycle, retirement, or marker mechanism** separating production models from scaffolds/experiments. The only signals remain per-config reads. Consequences: the all-models parametrized test fleet (conftest `ALL_MODEL_DIRS`) is inflated by non-production dirs; "what is real vs scaffold" is cognitive load with no machine answer; and cross-cutting changes (partition bumps, config-key migrations) touch experimental dirs indiscriminately. Still Tier 4 — no correctness impact, but a growing maintenance/comprehension cost. Exit direction (not applied): a maturity/marker convention (ties to ADR-017's proposed `maturity` axis) or a `models/experimental/` separation. **Tier recalibrated 4 → 3 during review-rr (2026-10-25):** the original Tier 4 rested on "incompatible spatial dimensions would cause a shape mismatch error" — a loud failure at ensemble aggregation. That premise still holds for the ensemble path but no longer bounds the entry. The 2026-07-20 broadening made this about ~30 unmarked experimental directories under fleet-wide sweeps, and `heavy_freighter` is now the control vehicle of a live measurement programme writing production-mode artefacts — C-171 records a 5-lesson rehearsal artifact being "SECURED" as though it were the measured run. An unmarked experimental directory that emits production-shaped output is a fleet-wide maintainability and coupling risk, not a single-model code-quality note. **Anchor of Cluster F** (no lifecycle for model directories), which until now had one member and it was not this one. |

---

### C-55 — Stale `xfail` marker on `test_datafactory_query_importable` produces xpass noise

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | A developer reviews CI output and sees an xpass warning for `test_datafactory_query_importable`, masking real xpass regressions |
| **Source** | falsify Round 3 (2026-06-04) |
| **Status** | Resolved |
| **Location** | `tests/test_bright_starship_readiness.py:29` |
| **Notes** | The `@pytest.mark.xfail` decorator on `TestF1_DatafactoryQueryDependency` was stale — `datafactory_query` is now installed. Removed the xfail; the test is environment-gated by the class-level `skipif(not shutil.which("conda"))`. See C-38. **Resolved 2026-06-04.** |

---

### C-56 — Override partition files become silently stale after annual bump

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | After an annual partition bump, the 8 PARTITION_OVERRIDE HydraNet models continue using pre-bump partition values |
| **Source** | falsify: bump completeness (2026-06-06) |
| **Status** | Resolved |
| **Notes** | **Resolved 2026-06-06:** Root cause was the ingester3 dependency — all 8 override files existed solely to avoid importing `ViewsMonth`. Removed ingester3 from all 83 files, replaced with inline `_current_month_id()`. Removed all `# PARTITION_OVERRIDE:` comment markers. Replaced with a programmatic `PARTITION_OVERRIDE = True` flag for legitimate research overrides (currently unused). The bump tool now updates all 100 files uniformly. See C-01. |

---

### C-57 — Regex parser matches comments instead of real dict in config_partitions.py

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A developer adds a comment like `# Old values: "calibration": {"train": (100, 200), "test": (201, 250)}` to a config_partitions.py file; the next bump silently writes new values into the comment and leaves the actual partition dict unchanged |
| **Source** | falsify: bump edge cases (2026-06-06) |
| **Status** | Resolved |
| **Location** | `tools/partitions/fileops.py:extract_values()` and `rewrite_values()` — regex `"calibration":\s*\{(.*?)\}` matches first occurrence |
| **Notes** | The regex matches the first occurrence of `"calibration": {` in the file. If that's in a comment, docstring, or dead code, `extract_values` reads wrong values and `rewrite_values` modifies the wrong location. No current file triggers this, but a single comment addition would cause silent corruption. **Tier 2 justification:** silent data corruption — the tool reports success while leaving the actual partition values unchanged. **2026-06-28 (pattern recurrence, config_meta):** the same comment-vs-real-dict regex hazard recurred *outside* the partition tooling — an ad-hoc model-cloning script (12 CM datafactory models) patched `config_meta.py` `regression_targets` with a `count=1` regex that matched a **commented** template line (`# "regression_targets": [...]`) ahead of the real key, leaving the real target wrong. Caught by per-model spot-check before commit (not shipped → C-57 stays **Resolved**). Confirms the pattern is general to **any regex-based config edit**: source models carry commented template key-lines in `config_meta.py`, so config-patching tooling must skip commented lines. Logged so future model-cloning/config tooling guards against it. |

---

### C-58 — `_load_canonical()` has no error handling for missing/corrupt partitions.json

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | `meta/partitions.json` is deleted, moved, or edited with invalid JSON; the bump tool prints a raw Python traceback instead of a helpful error message |
| **Source** | falsify: bump edge cases (2026-06-06) |
| **Status** | Resolved |
| **Location** | `tools/partitions/bump.py:_load_canonical()` |
| **Notes** | The function is two lines: `open()` + `json.load()` with no try/except. Missing file → `FileNotFoundError`. Corrupt JSON → `JSONDecodeError`. Missing keys → `KeyError` from `PartitionBoundaries.from_json()`. For annual critical infrastructure run by a maintainer, a raw traceback is a robustness failure. |

---

### C-59 — `write_atomic()` does not clean up temp files on `os.replace()` failure

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | `os.replace()` fails during a bump (permission error, disk full) after the temp file has been written; orphaned `.tmp` files remain in config directories |
| **Source** | falsify: bump edge cases (2026-06-06) |
| **Status** | Resolved |
| **Location** | `tools/partitions/fileops.py:write_atomic()` |
| **Notes** | Creates `NamedTemporaryFile(delete=False)` and calls `os.replace()`. No try/finally to clean up the temp file if replace raises. A failed run touching 100 files could leave up to 100 orphaned `.tmp` files. Low probability in practice (os.replace rarely fails on same-filesystem renames) but easy to fix with try/except around os.replace. |

---

### C-60 — Repo root and scripts/ mix operational tooling, scaffolding, and investigations with no structural separation

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A new contributor tries to understand the tooling layout and must read 8+ filenames at the root and 14+ files in scripts/ to distinguish operational tools from scaffold builders from investigation scripts |
| **Source** | falsify: tools organization (2026-06-07); **regression measured by falsify 2026-07-31** |
| **Status** | **Re-opened 2026-07-31, then re-closed with a guard** — it had been marked Resolved while regressing |
| **Trigger (added 2026-07-31)** | A tool is added to `tools/` without a directory named for its responsibility. The rule is documented in `tools/README.md`'s own opening line and nothing checks it, so the decay is invisible until someone counts. |
| **Location** | Originally: repo root (6 Python, 2 shell), `scripts/`, `tools/` (partitions only). **Regression:** `tools/` root — 2 loose files at the 2026-06-07 resolution (`__init__.py`, `audit_shell_health.sh`), **6** by 2026-07-31. |
| **Notes** | Violates CCP (catalog scripts change together but aren't grouped), CRP (4 unrelated responsibilities in one directory), and screaming architecture (flat layout requires reading every filename). Fix: organize into `tools/catalogs/`, `tools/scaffold/`, `tools/partitions/`; move investigation scripts to `investigations/`; move root shell scripts to appropriate locations. See ADR-011, C-01. **REGRESSED, AND THE REGISTER SAID OTHERWISE (falsify probe P4, 2026-07-31).** Counted at the resolving commit vs today: **2 loose files → 6**. All four additions post-date the resolution, three of them within five days — `audit_queryset_transforms.py` (2026-06-08), `check_credentials.py` (2026-07-27), `registry_to_env.py` (2026-07-28), `verify_committed.sh` (2026-07-31, added by the assistant during this very work). The entry read `Resolved` throughout. **The generalisable finding is not the mess but the measurement:** a structural rule stated in prose (here, `tools/README.md`'s "each subdirectory handles one responsibility") decays silently, and a register status is a claim about the past that nothing re-checks. **Re-closed 2026-07-31** by grouping on responsibility — `tools/credentials/` (`check_credentials.py`, `registry_to_env.py` — they change together, CCP) and `tools/audit/` (`shell_health.sh`, `verify_committed.sh`, `queryset_transforms.py` — read-only verification passes) — leaving only `__init__.py` at the root. **WET was preserved deliberately:** the three credential readers (`tools/credentials/*`, `tools/liveness/appwrite_api.py`) share zero code and were **not** unified; co-location is not consolidation. Two moved files carried `parents.parent` repo-root constants that silently rebased one directory deeper — caught by the test suite, fixed to `parents[2]`, and worth remembering as the standing cost of depth-counted paths. **Residual:** the count is still not machine-checked; a structural test asserting "no loose executable files at `tools/` root" would convert this entry from prose to a tripwire. Until that exists, expect the same drift. |

---

### C-61 — Fixture exclusion lists diverge across 3 locations

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A new fixture model is added to `_FIXTURE_ENTRIES` in `create_catalogs.py` but not to `_FIXTURE_NAMES` in `fileops.py` or `_FIXTURE_MODELS` in `conftest.py` — causing inconsistent catalog output, bump coverage, and test discovery |
| **Source** | repo-assimilation (2026-06-07) |
| **Status** | Resolved |
| **Location** | `tools/partitions/fileops.py:_FIXTURE_NAMES` (12 entries), `tools/catalogs/create_catalogs.py:_FIXTURE_ENTRIES` (12 entries), `tests/conftest.py:_FIXTURE_MODELS` (1 entry) |
| **Notes** | Three independent fixture exclusion sets. `_FIXTURE_MODELS` in conftest has only `fake_model` while the other two have 12 entries. The sets happen to not conflict currently because conftest uses `main.py` presence (not name) to discover models, so the extra 11 fixture names in the other lists are redundant there. But the naming inconsistency (`_FIXTURE_MODELS` vs `_FIXTURE_NAMES` vs `_FIXTURE_ENTRIES`) and the different cardinalities create confusion. Should be unified into a single source of truth. **2026-06-27 (#99 — completes a partial resolution):** the earlier fix introduced the canonical `meta/fixtures.json` for `fileops.py` + `create_catalogs.py`, but **`update_readme.py` was missed** — it still hardcoded `{fake_model, test_model, test_ensemble}` (3 of 12), so it would emit READMEs for the 9 synthetic fixture models it should skip. #99 repoints `update_readme.py` to load `meta/fixtures.json` (all three catalog/partition consumers now derive from one file) and hardens `test_bump_partitions.TestFixtureSetConsistency` — the prior check AST-matched a *set literal* and went **vacuous** once create_catalogs switched to JSON; it now source-checks that create_catalogs + update_readme load `fixtures.json` and hardcode no literal (negative-tested to fail loud). `conftest._FIXTURE_MODELS` stays `{fake_model}` **by design** — name-based exclusion for a different concern (test discovery uses `main.py` presence). |

---

### C-62 — No CIC for tools/partitions/ (partition bump tool)

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | A developer modifies `tools/partitions/domain.py` or `fileops.py` behavioral guarantees without a contract to verify against |
| **Source** | repo-assimilation (2026-06-07) |
| **Status** | Resolved |
| **Location** | `tools/partitions/` (3 modules, 37 tests, 3 falsification audits, but no CIC) |
| **Notes** | The partition tooling is the most thoroughly tested and audited component in the repo (37 unit tests, 3 falsification rounds, expert code review). But it has no Class Intent Contract documenting its guarantees, failure modes, or boundaries. The CIC sync check workflow (`cic_sync_check.yml`) cannot flag changes to this tool. Low urgency since the test coverage is strong, but the contract gap creates a documentation asymmetry with the other tools (all have CICs). |

---

### C-63 — Partition bump test files missing ADR-005 category markers

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | Test category analysis (red/beige/green distribution) reports inaccurate numbers because 4 test files (test_bump_partitions.py, test_falsify_bump_completeness.py, test_falsify_bump_edge_cases.py, test_falsify_bump_robustness.py) have no ADR-005 markers |
| **Source** | test-review (2026-06-07) |
| **Status** | Resolved |
| **Location** | `tests/test_bump_partitions.py`, `tests/test_falsify_bump_*.py` (3 files) |
| **Notes** | ADR-005 defines the red/beige/green taxonomy for test classification. The 4 partition bump test files (37 tests total) were written without category markers. Most are green (functional correctness) with some beige (structural compliance). The falsification verification tests could be marked green (they verify fixes). Low priority but creates a documentation gap in test distribution reporting. |

---

### C-64 — Zero red (adversarial) tests for all 9 tool modules

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A developer introduces a bug in tools/partitions/ or tools/catalogs/ that only manifests with adversarial input (corrupt file, permission error, concurrent execution); no red test catches it |
| **Source** | falsify: test category completeness (2026-06-07) |
| **Status** | Resolved |
| **Location** | `tests/test_bump_partitions.py`, `tests/test_catalogs.py`, `tests/test_scaffold_builders.py`, `tests/test_tooling_scripts.py` |
| **Notes** | **Resolved 2026-06-07:** 30 red tests now cover 8 of 9 tool modules. Partition tooling: 9 red (garbage input, partial structure, negative month_ids, missing return, missing section, negative bump, missing JSON key, non-iterable value, permission error cleanup). Scaffold: 3 red (github failure, without_directory_raises x2). Catalogs: 11 red (malformed markers, empty content, missing keys, non-list targets, empty model list, adversarial regex input). `build_package_scaffold.py` cannot be tested without `views_pipeline_core` — accepted gap. Also found: `_format_targets` crashes on non-string non-list input (TypeError) — characterized as red test. |

---

### C-65 — generate_features_catalog.py and update_readme.py have no core-functionality tests

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A developer modifies the main loop of `generate_features_catalog.py` or `update_readme.py` (model discovery, config loading, output generation); no test catches a regression in core behavior |
| **Source** | falsify: test category completeness (2026-06-07) |
| **Status** | Open |
| **Location** | `tools/catalogs/generate_features_catalog.py` (115 lines, 4 regex characterization tests only), `tools/catalogs/update_readme.py` (276 lines, 6 helper characterization tests only) |
| **Notes** | **Partially resolved 2026-06-07:** Added 11 functional tests for `generate_features_catalog.py`: 5 for `extract_columns_from_querysets()` (single file, dedup, loa extraction, empty dir crash, non-Python ignored) and 6 for `generate_markdown_table()` (valid markdown, headers, placeholders, row count, empty crash, queryset preserved). Found 2 bugs: empty dir crashes groupby (C-66), empty DataFrame crashes tabulate (C-67). `update_readme.py` orchestration remains untestable without views_pipeline_core — accepted. |

---

### C-66 — `generate_features_catalog.py` crashes on empty input (both functions)

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | `generate_features_catalog.py` is run against a model set that yields no queryset columns — an empty or non-Python directory, a filtered-to-nothing model list, or a fresh scaffold before any `config_queryset.py` exists |
| **Source** | test: catalog core tests (2026-06-07); merged with C-67 during review-rr (2026-07-31) |
| **Status** | Open |
| **Location** | `tools/catalogs/generate_features_catalog.py:72` (`extract_columns_from_querysets`), `:97` (`generate_markdown_table`) |
| **Notes** | Two crashes on the same empty-input path, one immediately downstream of the other. **(1) `extract_columns_from_querysets():72`** — builds an empty `columns_info` list, converts it to a column-less DataFrame, then calls `df.groupby(['column_name','loa'])`, which raises `KeyError` because those columns don't exist. Fix: early return when `columns_info` is empty. **(2) `generate_markdown_table():97`** — `tabulate(..., colalign=("center",))` assumes at least one data column; an empty DataFrame has zero, so it raises `IndexError`. Fix: skip `colalign` when `table_data` is empty, or return a header-only table. Both characterized as red tests (`test_empty_directory_crashes`, `test_empty_dataframe_crashes_tabulate`). **Merged with C-67 during review-rr (2026-07-31)** — same file, same session, same input condition, adjacent lines, and the second is only reachable through the first; two entries overstated the count without adding information. Member of **Cluster D** (catalog tooling is a script, not a program). **Backlog candidate:** mechanical fix, single file, Tier 4 — see the review-rr demotion list. |

---

### C-67 — `generate_markdown_table()` crashes on empty DataFrame *(merged into C-66)*

| Field | Value |
|---|---|
| **Status** | **Merged into C-66** (review-rr, 2026-07-31) |
| **Notes** | ID retained as a stub so existing cross-references resolve. Same file (`tools/catalogs/generate_features_catalog.py`), same empty-input condition, and reachable only through the C-66 crash path. Tracked together at C-66, Tier 4. |

---

### C-68 — `config_meta.py` is NOT documentation-only: pipeline-core's merge order makes it the winning source, and its own docstring says the opposite

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Reading `config_meta.py`'s docstring, or this entry's pre-2026-10-25 text, and concluding that a key declared there is inert — in particular, answering "is \<key\> on?" from `config_hyperparameters.py` and the library default without checking `config_meta.py` and the merge order. Also the original trigger: editing `regression_targets`/`level`/`algorithm`/`prediction_format` in `config_meta.py` expecting the edit to take effect. |
| **Source** | repo-assimilation (2026-06-09) |
| **Status** | Open |
| **Location** | `models/*/configs/config_meta.py`, `models/*/configs/config_hyperparameters.py`; **the merge order that decides it:** `views_pipeline_core/managers/configuration/configuration.py:460-492` (`_get_raw_combined_config`, and `_SAFETY_CRITICAL_KEYS`); the measured instance is `models/*/configs/config_meta.py` `"diagnostic_visualizations": True` across all 15 HydraNet models (`heavy_freighter`'s since `d3085f42`, 2026-04-28) |
| **Notes** | `config_meta.py`'s docstring states "modifying it will not affect the model, the training, or the evaluation." Yet several keys it declares — notably `regression_targets` — are also required as *operational* keys in `config_hyperparameters.py` (C-52 added `regression_targets` to 9 hyperparameter files for PFE participation). The same logical field thus lives in two files with opposite semantics: inert in meta, behavioral in hyperparameters. No test asserts the two copies agree, and no warning fires when a developer edits the inert copy. A change to the meta copy is silently ignored; a stale meta copy also misleads readers and the generated catalogs (`tools/catalogs/create_catalogs.py` reads `config_meta.py`). Low severity — no model-output corruption — but a maintainability footgun amplified across 90 models. See also C-52 (regression_targets added to hyperparameters), C-53 (stray `prediction_format` key leaked into hyperparameters during merge). **Tier recalibrated 4 → 3 during review-rr (2026-07-31):** "a change to the meta copy is silently ignored" across ~90 models is the *same defect class* as C-104 (Tier 2 — "the CI contract validates a key the runtime may not read"). A silently-ignored config edit is not a Tier-4 code-quality observation; it is the repo's signature failure mode. Held at 3 rather than 2 only because the wrong-edit here produces stale documentation rather than a wrong forecast. Member of **Cluster A** (declared-but-unenforced), with C-104 and C-85. ⛔ **2026-10-25 — THIS ENTRY WAS FACTUALLY INVERTED, and the inversion cost a working 300-lesson run.** `config_meta.py` is not documentation-only. pipeline-core composes a config from five sources **in order** — `partition_dict` → `config_hyperparameters` → `config_deployment` → `config_meta` → `_runtime_config` — with a plain `config[key] = value` per key (`configuration.py:460-492`). **`config_meta` is fourth, so it OVERWRITES `config_hyperparameters`**, and `_SAFETY_CRITICAL_KEYS = frozenset({"level", "run_type", "prediction_format"})` is the only set that warns — so for every key outside those three the override is **silent**. The file's docstring (*"modifying it will not affect the model, the training, or the evaluation"*) is therefore not merely stale: it is wrong in the direction that makes a reader stop looking. ⛔ **Measured instance, and what it cost.** `diagnostic_visualizations` is **absent** from every HydraNet `config_hyperparameters.py`, defaults **False** in `config_initializer.py:436`, and is **`True` in `config_meta.py` for all 15 HydraNet models**. Asked whether diagnostic visualisations were on, this seat read the default and the hyperparameters file, never opened `config_meta.py` or the merge order, and reported **OFF**. Simon killed a running, healthy control run on the strength of that statement. The viz had been on for every measured run behind the 202–272 min/n=3 figure. ⚠️ **Disconfirming evidence was present and ignored:** `/workspace/views-models/models/heavy_freighter/reports/plots` existed on the pod, and `models/*/reports/` tracks only `.gitkeep` — so that directory could only have been created at runtime by `VisualDiagnostics.__init__`. The filesystem said *on* while the reading said *off*. **Tier promoted 3 → 2 (2026-10-25).** The entry was held at 3 explicitly *"only because the wrong-edit here produces stale documentation rather than a wrong forecast"*. That premise is void: the key controls runtime behaviour on 15 models, the override is silent, and the realised harm was a destroyed GPU run plus a false statement to the operator. The asymmetry also runs both ways — **some** meta keys are genuinely inert (nothing reads them) and **some** are authoritative, and nothing in the repository distinguishes the two, which is C-157's shape at file granularity. Cross-refs: **C-175** (the diagnostic habit that produced the wrong answer — a default or a warning read as proof a knob is dead, without grepping for its consumer), **C-157** (a dead, missing or look-alike key is indistinguishable by reading), **C-104** (CI validates a key the runtime may not read — this is the mirror image: the runtime reads a key the reader believes inert), **C-168** (ROOT). |

---

### C-69 — `config_sweep.py` has zero test coverage and no validation of swept-parameter structure

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A developer edits a model's `config_sweep.py` and mistypes a swept parameter — e.g., `'values': [...]` written as `'value': [...]`, or a parameter name that does not match `config_hyperparameters.py` — then launches `--sweep`; the sweep runs but silently pins or ignores the parameter |
| **Source** | repo-assimilation (2026-06-09) |
| **Status** | Open |
| **Location** | `models/*/configs/config_sweep.py` (observed: `models/violet_visitor/configs/config_sweep.py`) |
| **Notes** | Unlike `config_meta.py` (`test_config_completeness.py`), `config_partitions.py` (`test_config_partitions.py`), and `config_hyperparameters.py` (C-05 ReproducibilityGate), `config_sweep.py` has no structural or semantic test. The current working-tree rewrite of `models/violet_visitor/configs/config_sweep.py` (a 128-line hand edit mixing `{'value': ...}` and `{'values': [...]}` entries) illustrates the exposure: a `values`→`value` typo silently converts a swept dimension into a fixed constant, and a parameter key that does not correspond to a hyperparameter is silently ignored by W&B. Failures are not loud — the sweep completes but explores the wrong space, wasting GPU/compute and surfacing a misleading "best" run. Affects anyone running sweeps. See also C-05 (HP presence validation — does not cover sweep configs), D-04 (static-analysis vs behavioral-execution test gap). |

---

### C-70 — `run.sh` environment-bootstrap logic duplicated across ~90 protected scripts

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | The planned conda→uv migration (see `reports/conda_to_uv_migration_*`) or any change to env-bootstrap logic requires editing the near-identical `run.sh` in every model/ensemble/api/extractor/postprocessor directory |
| **Source** | repo-assimilation (2026-06-09) |
| **Status** | Open |
| **Location** | `models/*/run.sh`, `ensembles/*/run.sh`, `apis/*/run.sh`, `extractors/*/run.sh`, `postprocessors/*/run.sh` (~90+ scripts) |
| **Notes** | Every model carries a near-identical `run.sh` that bootstraps a conda env, dry-run-checks `requirements.txt`, and invokes `main.py`. The bootstrap logic is duplicated rather than sourced from a shared script, so a change must fan out across all ~90 files — and these files are production infrastructure that must not be casually modified (operating constraint). C-39 already demonstrated the fan-out cost (79 shebangs corrected in one sweep); C-50 notes `run.sh` cannot be edited to fix the local-install path. The duplication is consistent with the project's accepted self-containment stance for configs (D-01), but unlike partition configs there is no `meta/`-style single source of truth or bump tool for `run.sh` — it is accepted-by-default rather than deliberately governed. Low severity (failures are loud, at bootstrap time), but a coordination cost that recurs on every infra change. See also D-01 (intentional config duplication is load-bearing), C-39 (shebang fan-out — resolved), C-50 (`run.sh` modification constraint). **2026-08-02 (expert-code-review):** the duplication now has a measured correctness cost, not only a migration cost — **C-119** records that the duplicated install gate decides a production install from a line count of pip's log, and **C-115** records a version boundary encoded in the duplicated `env_path` line. The fix belongs in the generator (`template_run_sh.py`, views-pipeline-core#384), not in ~131 copies; fixing copies is what let C-39 regress 24 times. |

---

### C-71 — violet_visitor diverged from trio parity on two axes at once (regression loss *and* posterior sample count)

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Someone runs or interprets a golden_hour↔stellar_horizon parity comparison assuming the viewser and datafactory trios are matched — but violet_visitor differs on **both** the regression loss (`hurdle_nb`, formerly `lognormal_nll`, vs `tobit` on the other five) **and** `n_posterior_samples` (**8** vs **16** on the other five) |
| **Source** | review (PR #116, 2026-06-09); merged with C-87 (review-diff 2026-06-18) during review-rr (2026-07-31) |
| **Status** | Open |
| **Location** | `models/violet_visitor/configs/config_hyperparameters.py` (`EXPERIMENT_IN_PROGRESS = True`; `loss_reg`, `n_posterior_samples`); the skip + documenting assertion at `tests/test_datafactory_parity.py::test_both_trios_use_same_loss`, `::test_constituent_posterior_samples_match`, `::test_violet_visitor_is_experiment_in_progress` |
| **Notes** | violet_visitor's regression loss was intentionally changed from `tobit` to `lognormal_nll` (Arm-1 hurdle experiment, magnitude_calibration dossier 2026-06-08, issue #85; commit 908d383). The viewser trio (pink_pirate, blue_stranger, violet_visitor) and datafactory trio (bright_starship, bold_comet, blazing_meteor) were designed to be loss-identical so golden_hour (viewser ensemble) and stellar_horizon (datafactory ensemble) could be compared apples-to-apples (the parity programme behind C-48). violet_visitor's divergence breaks that: a golden_hour↔stellar_horizon comparison now confounds the loss change with the data-source change. `test_both_trios_use_same_loss` previously asserted strict uniformity (`{"tobit"}`); it was updated (PR #116) to pin the expected diverged state (five `tobit` + violet_visitor `lognormal_nll`), so the divergence is explicit and any *further* drift is still caught. The risk is interpretive, not silent — but a reader unaware of the experiment could draw wrong parity conclusions. Revisit when Arm-1 concludes: either restore `tobit`, or promote the hurdle loss across the whole trio. See also C-48 (variable-variant parity — resolved), C-37 (forecasting parity divergence), C-44 (concat aggregation quality), C-69 (sweep config untested). **2026-06-12:** the divergence persists but the loss moved again: `lognormal_nll` → `hurdle_nb` (TruncatedNB body + weighted-BCE gate; ZINB epic views-hydranet#102, decision A). `test_both_trios_use_same_loss` pin updated in the same changeset. The parity caveat is unchanged: golden_hour↔stellar_horizon comparisons still confound the loss change with the data-source change. **Second divergence axis, absorbed from C-87 (review-rr, 2026-07-31):** violet_visitor's `n_posterior_samples` was cut **16 → 8** on 2026-06-16 as an **interim OOM workaround** — the eval stage OOMs at 16; 8 is gated by a "one run completes without the eval-stage OOM" check and is to be **restored to 16 once the OOM is fixed** (tracked as `views-hydranet C-116` / views-hydranet#124, outside this repo's register). Consequence: the trio sample-count parity invariant is broken on the same model that already broke the loss invariant, and golden_hour's expected concat total shifts 48 → **40** (16+16+8, see C-74). `test_constituent_posterior_samples_match` was updated in the same change to **pin** the intentional divergence rather than assert uniformity — mirroring the C-71 loss pin — so both divergences are explicit and any *further* drift still fails. **Why merged:** one model, one cause (deliberate single-model experiments run against a trio designed for parity), one closing action, and both axes already pinned by tests — two entries doubled the apparent risk without doubling the information. **Revisit when both experiments conclude:** restore `tobit` + `n_posterior_samples: 16`, revert both test pins, or promote the changes across the whole trio. Member of **Cluster E** (parity-programme drift). See also C-74 (golden_hour sample count), C-72 (the numerical fallout of the loss experiment), C-37 (forecasting-boundary parity), C-48/C-49 (trio parity). **2026-08-03 — mechanism changed from exact-value pin to truthful skip (Epic #242 S1.5, views-hydranet#255; resolves views-platform/views-models#254 + #297).** The root problem surfaced by #254/#297: violet_visitor is an **actively churning** R&D model — its committed `loss_reg` is a moving target (the committed value was `mse`, the parity pin expected `hurdle_nb`, and the working tree flickers), so pinning an exact value flickers red/green and was the **last red test blocking the dev→main release**. Fix: violet_visitor's config now declares `EXPERIMENT_IN_PROGRESS = True`; `test_both_trios_use_same_loss` and `test_constituent_posterior_samples_match` **skip** it while still pinning the other five (`tobit` / `16`), and a new `test_violet_visitor_is_experiment_in_progress` asserts the marker so the skip is never silent (fails loud with re-pin instructions if the marker is removed). This unblocks the release without committing a transitional value; the exact settled value is pinned at Epic #242 S3 #246 when the fleet moves to the `gated_NB` roster. Same change removed the **retired `body_mask` knob** from violet's committed config (→ `body_supervision: 'all'`, ADR-065 migration). **2026-08-03:** the parity pins now **skip** violet_visitor via an `EXPERIMENT_IN_PROGRESS` marker in its config rather than pinning a value that churns while the experiment runs; re-pin when the roster lands (Epic #242 S3 #246). This note lives in Notes and not in Status because the Status field is a controlled vocabulary that `tests/test_falsification_merge_readiness.py::test_open_count_accurate` parses — prose after `Open` silently drops the entry from the header count. |

---

### C-72 — violet_visitor predictions overflow to `Inf` under the Arm-1 `lognormal_nll` loss

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | golden_hour (or any consumer) is next run/aggregated against violet_visitor's calibration predictions while it runs the Arm-1 `lognormal_nll` loss — 46–63% of regression cells are `Inf` |
| **Source** | repo-assimilation + falsify (2026-06-09) |
| **Status** | Open |
| **Location** | `models/violet_visitor/configs/config_hyperparameters.py` (`loss_reg: lognormal_nll`, `loss_reg_sigma: 0.9`, `hurdle_threshold: 0`); artifact `models/violet_visitor/data/generated/predictions_calibration_20260609_051916/` |
| **Notes** | Verified directly: the 2026-06-09 calibration run has `Inf` in **63.5% / 59.4% / 46.4%** of `lr_sb_best / lr_ns_best / lr_os_best` cells (finite max 3.4e38 = float32 ceiling); the prior `tobit` run (2026-06-08) was clean (0 Inf, max ≈ 4365). Root cause: the lognormal inverse `exp(µ)` overflows float32. **Classification targets (`by_*`) are sane** — the breakage is regression-only. There **is** a signal (`tests/test_pfe_production_readiness.py::TestTransformUndoScale::test_no_inf[violet_visitor_calibration]` catches it) → Tier 2, not Tier 1. **Accepted as an active experiment**: the user has chosen to leave violet_visitor's loss as-is (issue #85, magnitude_calibration dossier, commit `908d383`); this entry documents the known state — it is **not** a request to change the model. `lognormal_nll` is a registered, valid loss in views_hydranet (`utils/utils.py:66`); this is purely numerical, not a registration issue. To make the experiment usable, tame the overflow (clamp/bound `µ` in `views_hydranet` `LogNormalFixedSigmaLoss`). Downstream: a fresh golden_hour run would ingest the Inf. See also C-71 (same change's parity impact), C-74 (golden_hour sample count), C-44 (concat aggregation). **2026-06-12:** Arm-1 (`lognormal_nll`) is superseded — violet_visitor switched to `hurdle_nb` (ZINB epic views-hydranet#102, decision A), removing the overflow-prone lognormal inverse from the active config. The Inf-bearing 2026-06-09 artifact remains on disk until a fresh hurdle-NB calibration run replaces it; keep Open until a clean artifact exists (the `test_no_inf` guard stays armed). **Tier recalibrated 2 → 3 during review-rr (2026-07-31):** the overflow-prone `lognormal_nll` configuration is **superseded** — the active config is `hurdle_nb`, so no future run can reproduce the `Inf`. What remains is one stale on-disk artifact, guarded by an armed test that fails loud if it is consumed. Residual cleanup, not live fragility. Member of **Cluster E** (parity-programme drift). |

---

### C-73 — Ensemble scaffold builder imports an unreleased pipeline-core symbol; CI installs core unpinned

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | CI (or any fresh `pip install views_pipeline_core`) resolves a released pipeline-core (PyPI 2.3.0 / tag 2.3.1) that lacks `template_config_modelset` — the 3 `EnsembleScaffoldBuilder` tests fail at import and the builder is unusable |
| **Source** | repo-assimilation + falsify (2026-06-09) |
| **Status** | Open |
| **Location** | `tools/scaffold/build_ensemble_scaffold.py:8`; `.github/workflows/run_tests.yml:19` (`pip install views_pipeline_core`, unpinned); `tests/test_scaffold_builders.py::TestEnsembleScaffoldBuilderDirectoryCreation` |
| **Notes** | `build_ensemble_scaffold.py` imports `template_config_modelset` from `views_pipeline_core.templates.ensemble`. That symbol exists only on pipeline-core `development` — in **no released/tagged version**: PyPI latest is 2.3.0; git tag 2.3.1 is malformed (its `pyproject` still says `version = "2.3.0"` and it also lacks the symbol). CI installs the package **unpinned**, resolving to 2.3.0, so the 3 scaffold tests `ImportError` and the builder is broken against any release. Real fix (no skip): cut a properly-versioned pipeline-core release shipping the symbol — HEAD is **137 commits ahead of 2.3.1** (dependency removals, signature/exception changes) → likely **minor/major, not patch**; run a cross-consumer smoke-import first; prefer a minimal release branch over 2.3.0 — then pin views-models CI + the scaffold path narrowly to it. (Templates already package via poetry-core — no `packages` directive — so adding `templates/{model,ensemble,package}/__init__.py` is robustness, not the blocker.) See also C-31 (upstream API breakage), C-42 (synthetic models on unreleased core branch). **Tier recalibrated 2 → 3 during review-rr (2026-07-31):** the failure is a loud `ImportError` at collection time in CI, not a silent wrong result. Held prominent despite the demotion because it is the **named concrete blocker inside Cluster C** and a constituent of C-80's standing red CI — fixing it is a prerequisite for the green-CI baseline, which is itself the precondition for every other signal in this register being trustworthy. Member of **Cluster C** (cross-repo dependencies have no released contract). |

---

### C-74 — golden_hour `concat` yields 12 posterior samples instead of 48

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A `PredictionFrameEnsembleManager` `concat` ensemble (golden_hour: 3 constituents × 16 samples) is aggregated and the output carries fewer samples than the sum of its constituents |
| **Source** | falsify (2026-06-09) |
| **Status** | Open |
| **Location** | `ensembles/golden_hour` (`aggregation: concat`); views-pipeline-core `PredictionFrameEnsembleManager` concat path; `tests/test_pfe_production_readiness.py::TestPFEEnsembleAggregation::test_aggregated_sample_count[golden_hour_calibration]` |
| **Notes** | golden_hour (concat, 3×16) should aggregate to **48** posterior samples; its calibration artifact (`predictions_calibration_20260603_135314`, June 3 — **predates** the violet_visitor Inf, so NOT caused by C-72) has only **12**. 12 is not a clean multiple of 48, so this is unlikely to be mere staleness of one constituent (that would give 16/32) — it points to a real defect in the concat path (samples dropped/sub-sampled rather than concatenated), which would **silently understate ensemble uncertainty**. Verify with a fresh run: 48 → it was staleness; still 12 → real concat bug to fix in views-pipeline-core. See also C-44 (concat CRPS quality), C-45 (ensemble `-t` cascade), C-46 (PFE classification targets). **2026-06-18:** the expected total changed from 48 to **40** — violet_visitor dropped to 8 samples (C-87), so golden_hour's constituents are now 16+16+8; factor this into the fresh-run check. **2026-06-26 (rusty_bucket work):** a control falsifies the "concat-path-wide" hypothesis — `synthetic_chant` (3×64, equal constituents) aggregates to exactly **192 = sum**, confirming PFE concat *does* concatenate the sample axis correctly (`prediction_frame_ensemble.py:99`). So golden_hour's **12** is a golden_hour-specific defect (its unequal 16/16/8 constituents + a stale June-3 artifact), **not** a views-pipeline-core concat bug and **not** a test mis-encoding: `_expected_ensemble_samples` correctly expects `sum(samples)` (verified and kept). The `test_aggregated_sample_count` failure is artifact-dependent — it **skips in CI** (fresh clone has no artifacts) and fails only on stale local artifacts. Fresh-run check still owed: rerun golden_hour → 40 = staleness; still 12 = a real golden_hour aggregation defect. **2026-07-31 — investigated; the fresh run is now the ONLY way to settle it, and the artifact was never self-consistent.** Two facts recovered from git and disk: (1) at the artifact's own timestamp (`predictions_calibration_20260603_135314`) all three constituents were configured at **`n_posterior_samples: 64`** — sum **192**, not 48 and not 40 — so the artifact's **12** did not match its *contemporaneous* config either; config drift since (64/64/64 → 16/16/8) explains none of it. (2) The inputs are gone: `pink_pirate` and `blue_stranger` have **no prediction artifacts on this machine at all**, and violet_visitor's are from 2026-07-27 — so the pooled result cannot be reconstructed or diagnosed from artifacts, only re-run. The likeliest mechanism given (1) and (2) is **C-85** — the ensemble loads each constituent's cached `y_pred.npy` by artifact timestamp with no config fingerprint, so golden_hour pooled whatever stale per-constituent npys existed on 2026-06-03, not what the configs declared. That makes this a probable *instance* of C-85 rather than an independent concat defect, consistent with the `synthetic_chant` control (3×64 → exactly 192) still passing today. **Test made truthful the same day:** `test_aggregated_sample_count` compared a frozen artifact against *live* configs and so failed permanently until someone re-ran; it now skips with the drift spelled out (`pink_pirate: 64 at artifact time -> 16 now; …`) and asserts normally whenever the compared value has **not** moved — deliberately keyed on the *value*, not on "the config file changed", so `synthetic_chant` (the C-74 control) stays live. Guard pinned by `TestArtifactStalenessGuard` against going vacuous. **Tier recalibrated 2 → 3 during review-rr (2026-07-31):** the original Tier 2 was set while the "PFE concat drops samples platform-wide" hypothesis was live. The 2026-06-26 `synthetic_chant` control **falsified** it (3×64 → exactly 192), narrowing this to one stale June-3 artifact on one ensemble, in a test that skips in CI and is not on any delivery path. Re-promote to Tier 2 **if** the owed fresh run still returns 12 — that would restore a real, silent uncertainty-understatement defect. See also C-90 (degenerate-mixture stand-ins), C-91 (sample-count adequacy), C-71 (the 16→8 change that moved the expected total to 40). Member of **Cluster E** (parity-programme drift). |

---

### C-75 — bright_starship datafactory readiness test is mis-scoped for CI (false red)

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | CI runs `test_bright_starship_readiness.py::TestF1` — it shells `conda run -n views-hydranet-env` for a workstation-only env absent in CI, erroring (`EnvironmentLocationNotFound`) instead of testing a CI-checkable contract |
| **Source** | repo-assimilation (2026-06-09) |
| **Status** | Resolved (2026-06-12) |
| **Location** | `tests/test_bright_starship_readiness.py::TestF1_DatafactoryQueryDependency` (class `skipif` only checks `shutil.which("conda")`, truthy in CI) |
| **Notes** | The test is a local pre-flight probe (per its docstring) but executes in CI because the `skipif(not shutil.which("conda"))` guard passes (CI has miniconda) while the named env `views-hydranet-env` does not exist → false red. Real fix (no skip): provision `views-datafactory` in the CI job and assert a real `import datafactory_query` in the CI interpreter, plus static contract checks (requirements declares it; descriptor shape; spec resolvable). Add the equivalent for shining_codex (closes C-41). See also C-38 (datafactory_query availability), C-55 (prior stale-xfail on this test — resolved). **2026-06-12: Resolved** (issue #122, HYBRID design decided with maintainer): the conda probe is now a workstation pre-flight that skips truthfully when the target env is absent (`_conda_env_path` basename-matches `conda env list --json`, probes via `conda run -p`); CI-meaningful coverage moved to static contract checks (requirements declares views-datafactory; queryset imports datafactory_query; generate() exists via AST) — static because the queryset imports datafactory at module level and no pinned views-datafactory release exists (C-73 lesson: no unpinned git deps in CI). Real-install CI check deferred to a tracked follow-up issue, conditional on a datafactory release. Guard sanity itself is pinned by `TestEnvGuardSanity`. |

---

### C-76 — `test_values_not_log_compressed` applies a false invariant to `ZeroModel`

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | The PFE log-compression test runs against a zero/constant baseline (e.g. zero_cmbaseline) and asserts `max>10`, which a correct all-zeros prediction can never satisfy |
| **Source** | repo-assimilation + expert-review (2026-06-09) |
| **Status** | Resolved (2026-06-12) |
| **Location** | `tests/test_pfe_production_readiness.py::TestTransformUndoScale::test_values_not_log_compressed` |
| **Notes** | The `max>10` heuristic (guarding against predictions left on `log1p` scale) is valid for learned-magnitude models but FALSE for `ZeroModel`, which correctly emits all-zeros (zero_cmbaseline max=0.0 → perpetual fail). Verified `locf_cmbaseline` (max 17412) and `average_cmbaseline` (max 4743) legitimately pass and MUST keep the guard — so the fix is to exclude **`ZeroModel` only** (keyed off `config_meta["algorithm"]`) and, better, assert `max==0 and min==0` for ZeroModel (a ZeroModel emitting nonzero is itself a bug). Local-only (CI has no prediction artifacts). A test-design correction, not a coverage skip. **2026-06-12: Resolved** exactly as described (issue #129) — ZeroModel branch asserts all-zeros, all other models keep the `max>10` guard. |

---

### C-77 — synthetic_chant README omits cross-pattern CRPS-inflation semantics

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | A reader interprets synthetic_chant's ensemble CRPS as prediction quality, unaware it reflects cross-pattern disagreement measured against models[0]'s actuals |
| **Source** | repo-assimilation + falsify (2026-06-09) |
| **Status** | Resolved (2026-06-12) |
| **Location** | `ensembles/synthetic_chant/README.md`; `tests/test_falsification_synthetic_runs.py::test_falsify_01_synthetic_chant_readme_documents_crps_inflation` |
| **Notes** | Genuine documentation gap (TDD-red test). Constituents use different synthetic patterns — `lucid_dream`=`vertical_stripe` (models[0] → supplies ground-truth actuals), `vivid_dream`=`horizontal_stripe`, `waking_dream`=`diagonal_gradient`; the ensemble evaluates all predictions against models[0]'s actuals, so CRPS (constituent 0.000/0.002/0.043 → ensemble 1.044) measures cross-pattern disagreement, not prediction quality. Real fix: document these facts in the README (mirror `ensembles/synthetic_chorus/README.md`). See also C-43 (synthetic_chorus order-dependency), C-42 (synthetic models on unreleased core). **2026-06-12 (root cause):** the documentation EXISTED — added 2026-05-26 (`8af868e`, the same commit that added the test) — and was deleted by the 2026-06-04 README regeneration (`243873a`); `tools/catalogs/update_readme.py` rebuilds READMEs from the scaffold, preserving only the `## Created on…` tail. Re-writing the docs without fixing the generator (C-78) just re-arms the failure — sequence with C-78. **2026-06-12: Resolved together with C-78** (issues #123/#130) — semantics restored inside a `<!-- manual -->` block, which the fixed generator now preserves. |

---

### C-78 — README regeneration silently destroys hand-written documentation

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | `tools/catalogs/update_readme.py` is run (manually or via `update_catalogs.yml`) against any model/ensemble README carrying manual content outside the preserved `## Created on…` tail |
| **Source** | session investigation (2026-06-12) |
| **Status** | Resolved (2026-06-12) |
| **Location** | `tools/catalogs/update_readme.py:125-135` (scaffold rebuild, `## Created on` regex tail-preserve); `.github/workflows/update_catalogs.yml` (automated path) |
| **Notes** | Verified incident: the synthetic_chant CRPS-semantics documentation added 2026-05-26 (`8af868e`) was deleted by the 2026-06-04 regeneration (`243873a`, "docs: regenerate model catalog tables and per-model READMEs") — the direct cause of the C-77 test failure and the first June 4 CI red. The generator rebuilds each README from `README_scaffold.md` and preserves only the `## Created on…` tail, so ANY hand-written section in any of the ~100 model/ensemble READMEs is silently destroyed on every regeneration — no diff review gate on the automated path, no error signal. Tier 3 (silent destruction of committed work product; affects every contributor who documents a model). Real fix: preserve-markers (e.g. a `<!-- manual -->` block) or regenerate only the generated tables, plus a regression test that a marked manual section survives regeneration (C-65: the tool currently has zero tests). See also C-77 (the wiped instance), C-65, C-36. **2026-06-12: Resolved** (issue #130) — `tools/catalogs/readme_preserve.py` extracts `<!-- manual -->…<!-- /manual -->` blocks from the old README and re-appends them after regeneration (wired into both loops of `update_readme.py`); regression tests in `tests/test_readme_preserve.py` (chips at C-65). |

---

### C-79 — Stale strict-xfail on fired chunky_bunny readiness tripwire keeps suite red

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | Anyone runs the local suite (or reads its output) while `test_target_transform_fix_is_released` still carries `@pytest.mark.xfail(strict=True)` — the XPASS registers as a hard failure and noise-trains readers to ignore red |
| **Source** | session investigation (2026-06-12) |
| **Status** | Resolved (2026-06-12) |
| **Location** | `tests/test_chunky_bunny_readiness.py::test_target_transform_fix_is_released` |
| **Notes** | The tripwire worked exactly as designed: it was armed 2026-06-09 against "published views-stepshifter lacks `target_transform`" and fired when views-stepshifter merged the mechanism to main on 2026-06-08/09 (`261ef6c`, PR #74 → main merge #76, released as 1.3.0). The strict-xfail marker is now stale and produces a permanent suite failure (same genre as resolved C-55). Fix: flip to a plain assertion. The two sibling tripwires remain LEGITIMATELY red and must stay armed: `test_per_model_envs_exist` (envs/views_stepshifter, envs/views_r2darts2 unprovisioned on this box) and `test_ensemble_uses_the_fixed_code_path` (validation env ≠ execution env, placeholder). I.e., the release precondition is met but chunky_bunny is NOT yet runnable via run.sh envs — the #117 dev-mode run tracker sidesteps this. See also C-55 (genre), issues #117, #114, views-stepshifter#55. **2026-06-12: Resolved** (issue #128) — xfail removed; the test is now a plain regression guard with a `skipif` when the sibling views-stepshifter checkout is absent (CI-safe, the C-75 lesson applied proactively). The two sibling tripwires remain armed. |

---

### C-80 — No green CI baseline since 2026-06-04 — new failures arrive invisible, merges proceed unvalidated

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Any PR is merged to development while run_tests.yml is red — the merge is structurally unvalidated and any NEW breakage it introduces is indistinguishable from the standing red |
| **Source** | session investigation (2026-06-12) |
| **Status** | Open |
| **Location** | `.github/workflows/run_tests.yml`; GitHub Actions history (last green: 2026-06-04 01:21) |
| **Notes** | Every run_tests.yml run since 2026-06-04 01:21 has failed (40/40 checked). The standing red is the union of C-73 (scaffold/pipeline-core skew, since June 5), C-75 (bright_starship env probe, structural), and C-77/C-78 (README wipe, June 4). Consequence observed this week: three independent NEW breakages (June 5 scaffold skew, June 8 chunky_bunny tripwire fire, June 9 zero_cmbaseline false invariant) accumulated unnoticed because red-on-red signals nothing, and PRs #116–#126 were all merged on red CI. Tier 2: structural fragility with a realistic, recurring trigger — every merge until CI is green again. Exit: resolve C-73 + C-75 + C-77/C-78 (tracked as the CI-green umbrella issue), then adopt the policy that development merges require green CI. See also C-28 (CI only checks last exit code), C-03 (integration tests not in CI). **The LOCAL half, found and fixed 2026-07-31:** the same harm existed on the developer's machine for a different reason — two tests returned a verdict on **local workspace state** rather than on code, so a normal `pytest` was red by construction. `TestF1_UncommittedWork` asserted the working tree was clean (perpetual trigger: any work in progress failed it) on the false premise that "uncommitted changes will be lost on merge" — a GitHub merge does not touch a local tree. `test_aggregated_sample_count` compared a frozen artifact against live configs (C-74). Both were **green in CI and red locally** — the C-75 class inverted, and invisible in CI precisely because a fresh clone has neither dirty files nor artifacts. Rewritten to their real invariants (dirty files that *overlap the incoming diff*; artifacts whose compared *value* has drifted), each with negative tests pinning the guard against going vacuous. Local suite is now **7416 passed, 0 failed**. Lesson for this entry's exit: "green CI" is a necessary but insufficient target — a suite can be honest in CI and useless on the machine where the work happens. |

---

### C-81 — `update_readme.py`'s top-level write-as-you-iterate loops crash mid-run and fire on import

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | `update_catalogs.yml` runs on a fresh checkout (where `ensembles/cruel_summer` and `ensembles/white_mustang` have no tracked `artifacts/`), or a local regeneration runs while a stray partial model dir sits in `models/` — the script crashes after rewriting an arbitrary prefix of READMEs |
| **Source** | session verification of PR #133 (2026-06-12) |
| **Status** | Open |
| **Location** | `tools/catalogs/update_readme.py` (both loops construct `ModelPathManager`/`EnsemblePathManager` with default `validate=True`; writes happen per-directory as iteration proceeds) |
| **Notes** | Observed live, three separate crash points: stray untracked `models/teenage_dirtbag` and `models/cool_cat` (partial dirs, no `artifacts/`), then tracked `ensembles/white_mustang` (no `artifacts/` in git; `cruel_summer` same gap — the C-32 `.gitkeep` backfill covered models, not these ensembles). `ModelPathManager` raises `FileNotFoundError` on a missing standard dir, killing the whole run. Because the script writes each README as it iterates (`iterdir()`, unsorted), a crash leaves an arbitrary subset regenerated — locally confusing; in the workflow the step fails (post-C-28 `set -e`), so catalogs go silently stale rather than partially committed. Fix directions: (a) construct path managers with `validate=False` (catalog generation is read-only on the dir structure) or per-entry try/except + end-of-run failure summary; (b) backfill `artifacts/.gitkeep` for cruel_summer/white_mustang (C-32 extension to ensembles); (c) iterate only git-tracked dirs so workstation strays can't break tooling. See also C-32 (root cause for the tracked gaps — Mitigated, recurrence here), C-28 (exit-code masking in this workflow — Resolved), C-65 (catalog tools untested), C-78 (manual-block preservation — Resolved; orthogonal fix in the same script). **Second failure path, absorbed from C-93 (review-rr, 2026-07-31): the same two loops sit at module top level with no `if __name__ == "__main__"` guard** (`update_readme.py:84` models, `:215` ensembles), so merely *importing* the module to reuse a helper or unit-test a function executes the entire regeneration as a side effect — and crashes on the first incomplete dir. Demonstrated during #99: `python -c "from tools.catalogs.update_readme import _FIXTURE_ENTRIES"` raised `FileNotFoundError` on `models/teenage_dirtbag/artifacts`. This is why #99's `TestFixtureSetConsistency` had to **source-read** the file with a regex instead of importing it — the canonical fixture value is not assertable by import, which in turn is why that check went vacuous once create_catalogs switched to JSON (see C-61). **Why merged:** identical location (the two top-level loops), one refactor closes both — wrap each loop in a `def main()` behind `__main__`, construct the path managers with `validate=False`, and collect-then-write instead of writing as you iterate. Splitting "it crashes when run" from "it runs when imported" implied two fixes where there is one. Member of **Cluster D** (catalog tooling is a script, not a program), with C-65, C-66, C-83. **2026-09-19 (#483):** a narrower guard now sits in the same script — the one `get_queryset()` call is isolated against pipeline-core 3.3.0's `ImportError` (their #514), with `tests/test_update_readme_survives_missing_data_client.py`. It does **not** touch this entry: `ModelPathManager(configs_dir)` at construction (`validate=True`) still aborts the job on an incomplete model directory before the queryset is reached, and the loops still run at import. Open as before; do not read #483 as mitigation. |

---

### C-82 — Manual blocks duplicate when a README also carries a `## Created on` section

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | A README processed by `update_readme.py` carries BOTH a `## Created on` section and a `<!-- manual -->` block, and a regeneration runs — the block is emitted twice and multiplies on every subsequent run |
| **Source** | falsify (PR #133 audit, probe P1, 2026-06-12) |
| **Status** | Resolved (2026-06-12) |
| **Location** | `tools/catalogs/update_readme.py` (Created-on capture `re.search(r"(## Created on.*)", …, re.DOTALL)`, both loops); interaction with `readme_preserve.merge_manual_blocks` |
| **Notes** | The Created-on regex captures from the heading to END OF FILE; merged manual blocks live at the end of the file, so they get swallowed into the captured created-section (re-inserted via `{{CREATED_SECTION}}`) AND re-appended by the merge → duplication, compounding per regeneration. Latent when found: no README the script processes had a Created section (test_model/test_ensemble are fixture-skipped; apis/ and postprocessors/ are not iterated). Wrong-output is duplication, not loss → Tier 4. **Resolved same day:** `readme_preserve.strip_manual_blocks()` added; both loops now run the Created-on capture on the stripped text (blocks extracted from the original first). Falsification stub `tests/test_falsification_readme_preserve.py` un-xfailed to a plain regression guard. See also C-78 (sibling failure mode — loss), C-81 (sibling failure mode — crash), C-65 (catalog tools untested — now partially chipped). |

---

### C-83 — `## Created on` sections are lost on the second regeneration (heading rename breaks recapture)

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | A README gains a `## Created on` section and `update_readme.py` runs twice — the first run renames the heading to `## Model Created on`, which the capture regex `(## Created on.*)` no longer matches, so the second run drops the section entirely |
| **Source** | falsify (PR #133 audit, bonus discovery while pinning C-82, 2026-06-12) |
| **Status** | Open |
| **Location** | `tools/catalogs/update_readme.py` (both loops: `re.search(r"(## Created on.*)" …)` followed by the `[:2] + " Model"` heading rewrite) |
| **Notes** | Pre-existing, unrelated to the C-78/C-82 fixes. The rename-then-recapture mismatch means any created-section survives exactly one regeneration — which likely explains why NO currently-processed README has one (they were silently eaten by successive catalog runs over time; only fixture/non-iterated READMEs retain theirs). Same content-loss family as C-78 but a different mechanism. Fix directions: match both headings (`(## (?:Model )?Created on.*)`) and stop re-prefixing if already prefixed, or stop renaming the heading altogether. Alternatively: deprecate the special-cased created-section in favor of the `<!-- manual -->` mechanism (C-78), which is rename-proof. See also C-78, C-82, C-65. |

---

### C-84 — Constituent wandb partition metadata diverges after a partial re-run, blocking the ensemble evaluation report

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A subset of an ensemble's constituents is re-run after a partition bump (or any config change) while the rest keep older runs — their *latest* wandb run configs then disagree, and `EnsembleManager … -e -re` aborts the report with `Partition metadata mismatch between models` |
| **Source** | execution incident (chunky_bunny re-aggregate, 2026-06-13) |
| **Status** | Open |
| **Location** | `views-reporting/views_reporting/templates/reports/evaluation.py:189-211` (reads `get_latest_run(...).config` per constituent and requires `{run_type: {train,test}}` to match across all); `views-pipeline-core/views_pipeline_core/modules/wandb/utils.py:358` (`get_latest_run`) |
| **Notes** | The report's consistency guard checks each constituent's **latest wandb run config**, NOT the on-disk prediction windows. On 2026-06-13, re-running elastic_heart (post #119 +12-month bump) then re-aggregating chunky_bunny crashed the report: 20 constituents' latest wandb run held `{test [457,504]}`, smol_cat held the pre-bump `{test [445,492]}`, and new_rules/revolving_door had no findable wandb project (silently skipped — only 21 of 23 are even checked). **The aggregation itself was correct** — all on-disk predictions (incl. the outlier) align at month window `[457,492]`, and the ensemble predictions + metrics (MSLE 0.634) were written fine; only the report HTML was blocked. So the guard fails on metadata provenance even when the data is sound. Recurring hazard: whenever constituents are run across a config change at different times, their newest wandb runs diverge and the report breaks until the laggards are re-run. Mitigations to consider: read partition from the saved prediction metadata (the actual data) rather than the latest wandb run; warn-and-skip a divergent constituent instead of hard-failing; or document that an ensemble report requires all constituents on the same config epoch. Workaround used: re-run the stale constituent (smol_cat, issue #141) so its latest wandb run logs the current partition. The new_rules/revolving_door "no wandb project" skip is a related latent reporting gap (constituents absent from the metadata check and baseline comparison). Cross-refs: C-56 (config-file partition staleness — different layer, resolved), C-43 (ensemble order-dependence), C-74 (golden_hour sample count). |

---

### C-85 — Ensemble silently reuses stale/wrong constituent predictions on `--saved`; config changes are ignored with no signal

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Any hyperparameter that changes a constituent's forecast **output** (sample count, seed, window) is edited, then the ensemble is re-run with `--saved` **without manually clearing** `models/<name>/data/generated/predictions_*` — the prior forecast is silently reloaded and the config change has no effect |
| **Source** | repo-assimilation (R8) + chunky_bunny incident (2026-06-13); **live incident + falsify audit (2026-07-20, rusty_bucket FAO delivery)** |
| **Status** | Open |
| **Location** | `views-pipeline-core/.../managers/ensemble/prediction_frame_ensemble.py:688` (`ts = path_artifact.stem[-15:]` — the cache dir is keyed on the **model artifact's** timestamp) and `:813` (`_load_or_generate_pf`: `if y_pred_path.exists(): load` — a bare existence check, **no config hash, no staleness test**); per-model `models/<name>/data/generated/predictions_{run_type}_{ts}/{target}/y_pred.npy` layout owned by views-models |
| **Notes** | The ensemble resolves each constituent's cached PredictionFrame (`y_pred.npy`) by the **fitted-model artifact's timestamp** and loads it if present, regenerating only when absent. Because a baseline artifact is **sample-count-agnostic** (it stores the history window; samples are drawn at predict time), editing `n_samples` does **not** change the artifact, its timestamp is unchanged, and the ensemble keeps loading the **same stale `y_pred.npy` written at the old sample count** — the config edit is silently discarded. **Demonstrated live 2026-07-20:** across four rusty_bucket ensemble attempts, config edits (128→32→16) never took effect on any constituent; the on-disk npy set was a silent **mix** of S=128 (bison, crane, fox, otter, robin) and S=32 (finch, heron, lynx), **zero at the configured S=16**, and the parent OOM'd loading 5×S=128×3 targets ≈ 18 GB every time regardless of config. Deleting the `predictions_*.parquet` files (the obvious artifact) did nothing because the loader reads `y_pred.npy`, not the parquet — the operator had **no signal** that the run was stale (the progress bar completed in 10 s = a load, not the ~2 min = regeneration, the only visible tell). **Why so silent — three independent layers all miss it (the C-104 pattern):** (1) *no guardrail* — the cache existence-check carries no config fingerprint, and the ensemble balance guard (#160) fires too late and only on *unequal* counts, not "all-equal-but-wrong"; (2) *knowledge* — the sample-count key is fragmented and the edited key can be a decoy (C-104); (3) *no test* — nothing asserts a loaded constituent `pf.sample_count` equals the current config. Exit: fingerprint the cache on the config values that determine output (invalidate/regenerate on mismatch), OR key `pf_dir` on a forecast-config hash rather than the artifact timestamp, plus a test that a config change forces regeneration. Interim operator rule: clear the whole `predictions_{run_type}_*` dirs after any output-affecting hyperparameter change. Tier raised 3→2 (2026-07-20): a realistic, common action — change a hyperparameter, re-run `--saved` — silently serves the wrong forecast with no error, and cost a full delivery night before detection. Cross-refs: **C-104 (sample-count key fragmentation + decoy validation — the "knowledge" leg)**, C-84 (epoch divergence at report time), C-74 (golden_hour sample count), C-44 (quality-blind aggregation), C-14 (the *producer* side of the same missing run identity — the cache is keyed on an artifact timestamp precisely because artifacts carry no run ID), and the silent-when-unenforced class (C-94 / C-95). *Note (review-rr 2026-07-31): this list previously read "C-116-class", a dangling reference — `C-116` is an ID in the **views-hydranet** register (see C-71), not this one.* Member of **Cluster A** (declared-but-unenforced). Also **Cluster H** (no identity separates current from left-over) — dual membership, following the C-90 precedent: the reused prediction is both an unenforced declaration and a stale artefact indistinguishable from a current one. |

---

### C-86 — Ensemble constituents can have incoherent / near-mono-family feature sets with no comparability check

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | An ensemble is assembled (via `config_modelset.py`) from constituents whose querysets share little — and the result is read as a coherent model family rather than a mix of disjoint feature experiments |
| **Source** | repo-assimilation (R9) + Hurdle-model investigation (2026-06-13) |
| **Status** | Open |
| **Location** | `models/*/configs/config_queryset.py`; `ensembles/*/configs/config_modelset.py` (no cross-constituent feature check) |
| **Notes** | The 6 chunky_bunny Hurdle constituents have feature sets ranging **29→79** with only **5 features common to all six**; several are near-mono-family — `fast_car` is **89% V-Dem** (slow country-year democracy indices, almost no conflict history), `twin_flame` is **94% topic/NLP**, `high_hopes` is pure conflict-history with no structural covariates. They are not a designed family with a shared backbone — they read as separate feature experiments that happen to share the Hurdle wrapper. Nothing in config or tests asserts cross-constituent feature comparability, so an ensemble can silently combine models built on disjoint, individually-questionable feature bases, making the ensemble's behaviour hard to attribute or reason about. Distinct from C-48/C-49 (viewser-vs-datafactory *cross-source* parity). Maintainability/interpretability risk, not a correctness fault → Tier 4. Cross-refs: C-44 (quality-blind aggregation), C-48, C-49. |

---

### C-87 — violet_visitor `n_posterior_samples` diverged from trio parity (16→8 OOM workaround) *(merged into C-71)*

| Field | Value |
|---|---|
| **Status** | **Merged into C-71** (review-rr, 2026-07-31) |
| **Notes** | ID retained as a stub so existing cross-references resolve. C-87 (sample count 16→8) and C-71 (regression loss `tobit`→`hurdle_nb`) are two axes of one divergence on one model, both caused by deliberate single-model experiments run against a trio designed for parity, both already pinned by tests, and both closed by the same action (conclude the experiments, restore parity, revert the pins). The 16→8 detail — including the `views-hydranet C-116` / views-hydranet#124 dependency and the golden_hour 48→40 consequence — is preserved in full inside C-71. Tracked at C-71, Tier 3, **Cluster E**. |

---

### C-88 — Reconciliation geography source was not derived from the data (silent country-ID corruption risk)

| Field | Value |
|---|---|
| **Tier** | 1 |
| **Trigger** | A reconciling ensemble (`reconciliation: "pgm_cm_point"`) is migrated from viewser to views-datafactory while the reconciliation wiring assumes viewser geography |
| **Source** | maintainer review of EPIC #172 |
| **Status** | Mitigated |
| **Location** | `reconciliation/composition.py` (`_derive_source`, `build_reconciler_for_run`), `reconciliation/source_detection.py`, `reconciliation/reconciler_factory.py` |
| **Notes** | viewser uses VIEWS `country_id`; views-datafactory uses `gaul0_code` (different ids, 0% overlap). The original reconciliation wiring (#172) **hardcoded `source="viewser"`** and never inspected the ensemble's data source — and the docstrings/ADR-014 *claimed* a derivation that was not implemented. A reconciling ensemble migrated to datafactory would have silently built viewser geography against `gaul0_code` data → plausible-but-wrong reconciled forecasts, no crash. **Mitigated (2026-06-26, EPIC #192 / S2 #194):** the geography source is now **derived** from the data — the `reconcile_with` CM partner's constituents (`source_detection.detect_ensemble_source`) — and **fails loud** if (a) the source has no registered provider (datafactory has none yet → clear crash, never a silent viewser fallback) or (b) the PGM ensemble and its CM partner disagree on source. All four reconciliation ensembles are viewser today (parity unchanged). Residual: the datafactory `gaul0_code` provider is not built (#196) — until then datafactory reconciliation fails loud by design. **2026-07-06 (#144 wiring constraint):** pipeline-core's `load_cm_frame` resolves the `reconcile_with` partner via `EnsemblePathManager` (`managers/ensemble/cm_forecast_loader.py:41`; `ensemble.py:59,65` → `_target="ensemble"` resolves under `ensembles/<name>/` only) — a plain **model** under `models/` cannot be a `reconcile_with` target (fails loud, `cm_forecast_loader.py:61-70`). Consequence for #144: the gaul0 CM label target must be an **ensemble of the 12 CM datafactory models** (matches the maintainer's stated intent: "one of these or an ensemble of these"), or the loader must be generalized to accept a model path. Noted on issue #144. See C-40 (generate() contract), C-49 (viewser↔datafactory geography divergence), C-51 (pipeline-core `get_data` hardcodes viewser). |

---

### C-89 — Reconciliation wiring depends on the phasing-out viewser/pandas substrate

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | viewser / pandas / the custom `_PGDataset`/`_CDataset` are removed (the views-datafactory + views-frames migration) before reconciliation is re-platformed |
| **Source** | maintainer review of EPIC #172 |
| **Status** | Open |
| **Location** | `reconciliation/viewser_country_mapping_provider.py` (viewser + pandas fetch); views-pipeline-core `modules/reconciliation/adapter.py` + `data/handlers.py` (`_PGDataset`/`_CDataset`) |
| **Notes** | The reconciliation geography is fetched via a viewser `Queryset` returning a pandas frame — viewser and pandas are both being phased out for views-datafactory / views-frames. The `reconciliation/` package does **not** use the old custom `_PGDataset`/`_CDataset` directly (it is frames-native + numpy), but the reconciliation *flow* still rides them via pipeline-core's `reconcile_datasets` adapter — those custom dataframes were never sanctioned and do not scale to global PGM with posterior samples. The viewser provider carries a `TRANSITIONAL` comment. **Resolution path:** the per-source provider port (C-88 fix) lets a views-datafactory provider replace the viewser one (one file, #196). **2026-06-26 (#191, Epic 11):** the reconciler-algorithm relocation has **landed** — the concrete is now the frames-native, published `views_frames_reconcile` sibling (was `views_postprocessing`; ADR-023, PyPI v1.7.0), declared `views-frames>=1.7.0` in the reconciling ensembles. The residual is narrower: the viewser+pandas *geography fetch* in `ViewserCountryMappingProvider` (retired by #196) and the reconciliation flow still riding pipeline-core's `_PGDataset`/`_CDataset` adapter. Not a correctness risk today (viewser is the current source) — a structural/migration risk. **2026-09-19 (#485):** the 31 r2darts2 models moved to `views-r2darts2[manager]>=0.2.3` — NOT because the pandas lock lifted. viewser 6.6.4 (what pipeline-core 3.3.0 resolves) still caps `pandas<2`; r2darts2 0.2.3 sidestepped its own conflict by pinning `darts==0.40.0` / `pandas<2` again, and pipeline-core 3.3.0 widened wandb. The substrate this entry describes is unchanged; #473 carries the clock (Python 3.11 EOL 2027-10-31). |

---

### C-90 — rusty_bucket pools 8 identical baseline stand-ins (degenerate mixture until real constituents land)

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | `rusty_bucket` is promoted out of `deployment_status: shadow`, added to `monthly_run.sh`, or its forecasts are delivered to FAO via `un_fao`, **before** the 8 `temporary_*` clones are replaced by the real ~8 global HydraNets (#146) |
| **Source** | review-diff / register-risk (2026-06-26) |
| **Status** | Open |
| **Location** | `ensembles/rusty_bucket/configs/config_modelset.py` (8 `temporary_*` constituents); `models/temporary_{otter,robin,finch,heron,lynx,bison,crane,fox}/` |
| **Notes** | `rusty_bucket`'s 8 constituents are identical clones of the `heavy_strider` global-land baseline — interim stand-ins (#143/#146). PFE concat pools them to 8×128 = 1024 draws, but because all 8 are identical the pooled distribution is **degenerate** — statistically equivalent to one `heavy_strider` resampled, not a genuine 8-model mixture. The interim state is documented (README, `config_modelset` docstring, `deployment_status: shadow`, the non-blocking sample-count report) and is intentional: it validates the pooled-draw machinery at the correct global-land shape, not forecast quality. The risk is purely if `rusty_bucket` is run/delivered as a real forecast before #146 swaps in the diverse HydraNets — FAO would receive a single-baseline forecast that is, in the data itself, indistinguishable from a real ensemble. Retired by #146. Distinct from C-44 (heterogeneous quality dilution) — this is *homogeneous* degeneracy. See also C-44, C-91, #143/#146. Member of **Cluster B** (delivery machine has no closed loop) and **Cluster F** (no lifecycle for model directories). |

---

### C-91 — 128 posterior samples may be too few for stable HDI tails in production FAO delivery

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A concat ensemble (`rusty_bucket` or successor) is shipped to **production** FAO delivery at its current pooled depth and the summarizer (views-frames#89) computes 90/95% HDI / credible-interval bounds from the pooled draws. **The numbers in this entry were 128/constituent and 1024 pooled; both are now 8× smaller** — see Notes. |
| **Source** | register-risk (2026-06-26, ADR-015) |
| **Status** | Open |
| **Location** | `ensembles/rusty_bucket/configs/config_hyperparameters.py` (`expected_samples_per_model: 16`, i.e. the **produced** D×K width per ADR-015 §6); `docs/ADRs/015_posterior_sample_count_standard.md` §1, §6 |
| **Notes** | ADR-015 sets 128 as the *integration-period* per-model sample standard. For zero-inflated, right-skewed, heavy-tailed conflict posteriors, tail quantiles (90/95% HDI bounds) estimated from 128 draws are noisy; the pooled total (8×128 = 1024) helps, but per-constituent resolution still bounds tail stability. Fine for integration/shadow; ADR-015 explicitly flags revisiting (512–1024 per constituent) before production. The summarizer redesign (views-frames#89) is the consumer that should specify the required draw count. Not silent corruption — the draws are correct, only the tail estimates are noisy. **Restated 2026-08-10 (Epic #242 S4): the entry described a configuration that no longer exists, and had done for three weeks.** Two independent changes moved the numbers, neither of which updated this entry. (1) On **2026-07-20** the per-constituent count was thinned **128 → 16** because the full-S run peaks at ~28.6 GB and does not fit production hardware (recorded only in a comment in `ensembles/rusty_bucket/configs/config_hyperparameters.py`, cross-ref C-99). (2) ADR-015 **§6** now defines the contract's unit as the **produced** width D×K = `n_posterior_samples × n_head_samples`, not D alone. So the live figures are **16 produced per constituent** (D=4 × K=4 for the Epic #242 roster) and **8 × 16 = 128 pooled** — exactly one eighth of the 128/1024 this entry was written against, and *below* the per-constituent floor it was raised to warn about. **The concern is therefore strictly sharper than when it was filed, not resolved.** The 512–1024-per-constituent revisit ADR-015 flags before production is now further away, and the binding constraint is memory, not preference — so closing this needs either a summarizer that states its required draw count or a run that fits the hardware, not a config edit. **What this cost:** nothing detected the drift. The register is the artifact that is supposed to hold this, and a number inside prose is invisible to every test in the repo — cross-ref C-113, C-127 (the delivery map's counts are asserted precisely because prose numbers rot silently). See also C-90, C-44, C-99. |

---

### C-92 — `importorskip` on cross-repo deps converts breakage into silent skips

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A test `pytest.importorskip("<cross-repo module path>")`s a dependency, and that module is later **moved, renamed, or deleted** upstream — the test then silently SKIPs instead of failing, so a real regression ships green |
| **Source** | review-diff / register-risk (2026-06-26, PR #212) |
| **Status** | Mitigated (reconciliation suite) — pattern open elsewhere |
| **Location** | `tests/test_reconciliation_{composition,factory,e2e}.py` (fixed); same pattern at `tests/test_scaffold_builders.py:160,216,284` (`importorskip("views_pipeline_core")`) |
| **Notes** | The reconciliation tests `importorskip`-gated on `views_postprocessing.reconciliation` and `views_pipeline_core.domain.reconciliation`. Both moved/deleted upstream (vpp C2 deleted the former; pipeline-core #237 split the latter into `domain.reconciliation_port`), so **two real breakages went undetected** — `importorskip` turned the missing modules into SKIPs, CI stayed green, and the tests that validate the reconciler wiring were no-ops until #206/#212 caught it. **Mitigated (PR #212):** the guards were repointed to the live modules, and the reconciliation *package* now hard-imports `domain.reconciliation_port` at load, so a future upstream move ERRORs at collection (loud) rather than skipping. **Residual:** the pattern persists at other `importorskip` sites — prefer a hard import (declared deps should fail loud) or a fixture that asserts the dep is present, over `importorskip`, for any dependency that is a *declared* requirement rather than a genuinely optional one. See also C-42 / C-31 (cross-repo coupling), C-89 (reconciliation substrate). Member of **Cluster G** (a check out of scope returns a clean verdict). |

---

### C-93 — `update_readme.py` runs the full catalog regeneration at module import time *(merged into C-81)*

| Field | Value |
|---|---|
| **Status** | **Merged into C-81** (review-rr, 2026-07-31) |
| **Notes** | ID retained as a stub so existing cross-references resolve. C-93 (the loops fire on import) and C-81 (the loops crash mid-iteration leaving a partial rewrite) are two failure paths of one structural defect — **two write-as-you-iterate loops at module top level with no `__main__` guard** — closed by one refactor. Tracked together at C-81, Tier 3, **Cluster D**. |

---

### C-94 — Datafactory `country_month` aggregation sums intensive features silently; CM datafactory models must stay count-only

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A developer adds an intensive feature (a V-Dem index, most WDI rates) to a `country_month` views-datafactory model's `config_queryset.py` feature set — e.g. the 12 `{warring,ravaging,roaming}_{mage,cleric,fighter,thief}` CM models — before datafactory ships per-feature weighted-mean aggregation |
| **Source** | session investigation (2026-06-28, 12 CM datafactory model scaffold) |
| **Status** | Open |
| **Location** | views-datafactory `src/datafactory_adapters/grid_to_country_month.py` (`.groupby(["month_id","country_id"]).sum()` on all features); consumed by views-models `models/{warring,ravaging,roaming}_{mage,cleric,fighter,thief}/configs/config_queryset.py` (`loa: "country_month"`) |
| **Notes** | `grid_to_country_month` aggregates grid→country by **summing every feature**. Correct for **extensive/count** features (`ged_*_best`, `acled_*` — the minimal UCDP set these 12 CM models use), **wrong for intensive** features (indices/rates: V-Dem, most WDI) — summing an index across a country's grid cells is meaningless. The adapter only **warns** for known intensive prefixes (`_INTENSIVE_PREFIXES = shdi/healthindex/edindex/incindex/vdem_/ghs_built_`) and **still sums them**; critically, **WDI is in neither the intensive list nor the extensive (`ged_/acled_`) list, so WDI rates are summed with NO warning** — pure silent corruption of model inputs. datafactory **ADR-040 explicitly defers** intensive aggregation ("weighted average, not sum") to a future ADR. **Why this is a views-models concern:** the 12 CM datafactory "label" models (the gaul0 `reconcile_with` target for the reconciling rusty_bucket clone #144 — see project memory) are deliberately scoped to UCDP **counts only**; that constraint is **load-bearing, not stylistic**. Adding V-Dem/WDI before the datafactory per-feature weighted-mean aggregation lands feeds silently-wrong summed indices into training. Tier 2: structural fragility, clear trigger (add an intensive feature), **silent** failure (no error; meaningless values) — read with Tier-1 caution. Exit (datafactory-side): a per-feature aggregation registry (sum vs weighted-mean + a population/area weight) + fail-loud on unclassified features; then these models can gain richer features safely. The minimal-UCDP draft (branch `feature/datafactory-cm-r2darts-models`) carries this constraint as a comment in each `config_queryset.py`. See also C-48 (ged variant parity), C-13 / C-44 (quality-blind aggregation), C-89 (viewser→datafactory migration substrate). **2026-07-06 (ADR-048 landed — exit condition NOT met on our path):** datafactory shipped declared `feature_agg_types` (registry-declared extensive/intensive/static, per-feature; intensive-at-CM now **raises**) — the fail-loud exit this entry asked for. **But** the remote-zarr loader hardcodes `feature_agg_types=None` (`datafactory_query/dataset.py:223` — never read from zarr attrs, no HTTP fetch of `feature_agg_types.json`), and with `None` the ADR-048 block is skipped (`grid_to_country_month.py:127`) → the **old silent-sum behavior persists for every remote consumer**, which is what all views-models datafactory models use (`DEFAULT_REMOTE.zarr_url`). Re-assembling the store does not help — no code path consumes agg types remotely. Status stays **Open**, narrowed to the remote path; exit = datafactory wires agg types into the zarr/remote backend (flagged to maintainer, datafactory-side). |

---

### C-95 — r2darts `feature_scaler_map` silently skips missing/unmapped features; stale hardcoded maps rot without signal

| Field | Value |
|---|---|
| **Tier** | 1 |
| **Trigger** | A model's queryset feature set changes (or a config is cloned onto a model with a different feature set) without updating its `feature_scaler_map` — the map's stale names are silently ignored and any unmapped-but-present feature trains **unscaled** when `feature_scaler` (the default) is `None` |
| **Source** | expert-code-review (2026-07-05, 12-CM-model finalization) |
| **Status** | Open (root cause upstream; the 12-clone incident instance is fixed in this changeset) |
| **Location** | views-r2darts2 `views_r2darts2/transformers/feature_scaler_manager.py:73-74` (`_assign_default_scaler` early-returns when `default_scaler` is None → unmapped features get **no scaler, silently**) and `:121-124` (map columns absent from the data are silently intersected away: `if not feature_indices: continue`); incident instance: the 12 `models/{warring,ravaging,roaming}_{mage,cleric,fighter,thief}/configs/config_{hyperparameters,sweep}.py` (fixed → global `feature_scaler` chain); latent: the 4 source models' maps (`smol_cat`, `elastic_heart`, `new_rules`, `revolving_door` — currently correct for their viewser querysets) |
| **Notes** | **Demonstrated live:** the 12 CM datafactory clones inherited `smol_cat`'s ~44-name `feature_scaler_map` (viewser columns: `lr_vdem_*`, `lr_wdi_*`, `lr_topic_*`, decay/tlag/splag) against a 3-column datafactory frame. Net effect: `warring_*` (target sb; covariates ns+os both in the map) would train **accidentally correctly**, while `ravaging_*`/`roaming_*` (8 of 12) would train with their `lr_ged_sb` covariate **raw/unscaled** next to an asinh-scaled peer — silently degraded models, no error, no log line. No gate catches it: the ReproducibilityGate "Boundary Handshake" audits the dataframe **against its own columns** (`views_dataset_darts.py:31-35`, `expected_features=self.features` = df-derived), so map staleness can never fail it. **Fix (this changeset, Option C per expert-code-review):** delete the map in all 12; global `"feature_scaler": "AsinhTransform->MaxAbsScaler"` (same transform as the tuned sources, zero hardcoded names — ADR-013; chain strings verified supported, `scaler_selector.py:180`, applied globally at `darts_forecaster.py:178`). **Residual:** (a) the upstream silent semantics remain — file a views-r2darts2 issue asking for warn/fail-loud on unmapped features when the default scaler is None, and on map names absent from the data; (b) no test anywhere cross-checks `feature_scaler_map` names against a model's actual feature set — any future map user re-enters this trap; (c) the 4 source models' maps are correct today but rot the same way if their querysets change. Tier 1: silent training-input corruption with no error signal, demonstrated on 8 of 12 models, caught only by review. See also C-57 (comment/config drift class), C-94 (same silent-when-unenforced pattern, datafactory side). **Cluster G** (added 2026-10-26, #610) — the scaler map's input excludes the features it cannot resolve, so a stale name is silently ignored and an unmapped-but-present feature trains **unscaled**. The highest-tier member of the cluster. |

---

### C-96 — Validation test window (505–552) extends past last observed UCDP month; zero-filled actuals corrupt validation metrics

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A datafactory-sourced model is evaluated on the **validation** partition (test 505–552) and its metrics are trusted while the live store's `last_valid_month_id` < 552 — the tail months' "actuals" are zero-fill, not observations |
| **Source** | session investigation (2026-07-02, datafactory deep-dive for the 12 CM models) |
| **Status** | Mitigated (current partitions verified covered; re-arms on every partition bump) |
| **Location** | views-datafactory assembled store (grid spans months 109–564; UCDP Annual v25.1 observed through ~month 540 = Dec 2024, `datafactory_harvester/sources/ucdp_annual.py:57`); warning-only guard at `datafactory_query/dataset.py:483-495`; consumers: every views-models datafactory model's `config_partitions.py` validation partition (canonical test 505–552, `meta/partitions.json`) |
| **Notes** | The store's grid dimension covers the full validation window, so nothing errors — but `ged_*` values after the last observed UCDP month are **zero-padding, not observed zeros**. Evaluating against them silently rewards predicting zero for 2025 and contaminates CRPS/MSLE comparisons. `load_dataset` emits a `UserWarning` ("exceeds last observed data month") reading the live `.zattrs` `last_valid_month_id` — a signal, but one that survives only in run logs; the metrics themselves carry no marker. Calibration (test 457–504 = through Dec 2021) is fully observed and unaffected — pre-merge smoke runs for the 12 CM models use calibration for this reason. Same class of issue the un_fao delivery solved producer-side with `_clip_observed_history` (vpp S2/C-26); model evaluation has no equivalent clip. **Verified 2026-07-06:** live `get_last_valid_month_id()` = **558** — the current validation window (505–552) is **fully observed** (datafactory's harvest-freshness work extended coverage past UCDP v25.1's month 540). Current partitions safe → Status Mitigated; this entry is the standing tripwire and **re-arms whenever a partition bump outpaces observed coverage** — re-check the live value at every bump. Mitigation directions: evaluation-side clip of the test window to `min(test_end, last_valid_month_id)`, or a fail-loud gate when a partition's test window exceeds observed coverage. **Tripwire automated 2026-07-19 (epic #238 S2):** `python -m tools.liveness.datafactory_input` derives the requirement from `meta/partitions.json` at run time and compares it to the live `last_valid_month_id` — `INPUT_STALE` (exit 1) exactly when this entry's trigger fires; run it at every partition bump. See also C-01 (partition bump machinery), C-94 (datafactory silent-behavior class), the `# PARTITION_OVERRIDE` entries (month-boundary drift). |

---

### C-97 — Delivery selection is by recency, not identity: consumers pull "newest forecast in the bucket" and hope

| Field | Value |
|---|---|
| **Tier** | 1 |
| **Trigger** | Any second producer uploads to the shared PROD_FORECASTS bucket before a consumer pulls — including the scripted order of `monthly_run.sh` itself (4 ensembles upload, then un_fao pulls "newest", which is now one of those 4, not its configured `rusty_bucket` → the "sequencing trap": the scripted chain should fail at the FAO step every time, statically read; unconfirmed live because no full chain has been observed) |
| **Source** | delivery-machine map (2026-07-19) + expert-code-review adjudication |
| **Status** | **Resolved (2026-07-28)** — run 0 flipped the FAO path onto the wire: the manifest-addressed run `rusty_bucket_forecasting_20260727_095355` is what faoapi serves, resolved by identity, not recency. Was Mitigated 2026-07-20 pending exactly this event. |
| **Location** | vpp `unfao/managers/unfao.py` legacy reader (now `LEGACY_FORECAST_FILTERS = {"category":"forecast","type":"ensemble"}` — the §11.4 Hop-A guard, no longer the unfiltered `:106` newest-wins); `monthly_run.sh` (upload order); every future consumer inherits the pattern until on the wire |
| **Notes** | The **legacy** interchange's addressing model was **recency-as-identity**: a consumer asks for "the newest thing anyone uploaded" and post-hoc identity-checks it (`unfao.py:119`, S3/C-25). Expert adjudication (Kleppmann/Hickey): a **skeleton defect** — time-of-upload complected with identity because the artifact's identity triple (ensemble, month, version) was never reified in addressing. **Update 2026-07-20 (seat review §3.2):** the "no name filter" claim is stale — the ADR-013 §11.4 transition guards now type-pin both selectors (Hop-A vpp PR #99 → `type="ensemble"`; Hop-B faoapi PR #200 → `type="model"`), so a contract artifact can no longer be grabbed by a legacy selector, and vice versa. More fundamentally, **the ADR-013 wire's single per-run manifest (uploaded last = commit marker; §4.2/§4.3) IS the deterministic-addressing exit this entry demanded** — the consumer resolves "the latest *manifested* run" by identity, not raw recency; the identity guard demotes to defense-in-depth exactly as prescribed. The full wire is merged on the producer/ACL side but **not yet live** (faoapi consumer #100 S2–S6 unbuilt; upload interlock holding), so recency-as-identity persists on the *legacy* path until run 0 flips to the wire — hence Mitigated, not Resolved. Must NOT be preserved per review verdict. See also C-88 (identity coherence), C-94/C-95 (silent-when-unenforced), C-98 (dual store), C-100 (config-vs-reality). **Closed 2026-07-28 (status corrected during review-rr 2026-07-31):** this entry's own stated exit condition — *"recency-as-identity persists on the legacy path until run 0 flips to the wire"* — **was met**. Run 0 delivered a manifest-committed run to `unfao_bucket` and faoapi resolved and served it by identity (`run_id: rusty_bucket_forecasting_20260727_095355`, `mode: wire`), replacing the previously-served `orange_ensemble`. The identity guard is now defense-in-depth exactly as prescribed. The entry sat `Mitigated` for three days after its condition was satisfied — a reminder that condition-gated statuses need a closing pass. **What did NOT close with it:** the *serving* hop's silent last-good fallback (C-111) and the unpaginated-listing defect that exercised it (C-109) — deterministic addressing solved *which* run to fetch, not *whether the fetch succeeded visibly*. Member of **Cluster B** (delivery machine has no closed loop). |

---

### C-98 — Two network stores receive every forecast; no declared system of record

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Any consumer/tool reads the legacy views-forecasts store while another reads the Appwrite bucket for the same month (or one of the two saver uploads partially fails) — the two "truths" diverge with no detection mechanism |
| **Source** | delivery-machine map (2026-07-19) + expert-code-review |
| **Status** | Open |
| **Location** | pipeline-core `savers.py:159-191` (`ViewsForecastsSaver` → legacy store, via the C-47 list-in-cell conversion) + `savers.py:122` (`AppwriteSaver` → `APPWRITE_PROD_FORECASTS_*`, `prediction_store.py:14-22`); both fire on every `--prediction_store` run |
| **Notes** | Dual-write with no declared authority is an unresolved migration wearing architecture's clothes: which store is *the* forecast is undefined by construction, drift is unobservable, and the legacy leg still rides the OOM-prone list-in-cell DataFrame path (C-47). Expert disagreement D-β on timing: declare the authoritative store now (Kleppmann) vs after ground-truth observation of what consumers actually read (Beck) — either way it **must be decided**; the undeclared state must NOT be preserved. Exit: one store becomes the system of record; the other is demoted to explicit export or retired. **Observability (2026-07-19, epic #238):** both stores are now independently observable — `tools/liveness/appwrite_store.py` (Appwrite shelf) and `tools/liveness/vpn_store.py` (legacy gjoll, VPN-truthful) — so drift between the two "truths" is at least *visible* on demand (first live read showed exactly the split this entry predicts: Appwrite newest 2026-06-29 while the July 15 wandb runs uploaded nowhere). The instrument does not decide authority; the exit stands. See also C-47, C-97. |

---

### C-99 — Monthly production has no home and no heartbeat: an informal laptop rotation with no missed-month signal

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A month's run is skipped, half-completes, or fails on whoever's laptop was running it — nothing detects it; consumers (the classic store, a partner delivery) silently receive nothing |
| **Source** | maintainer ground truth (2026-07-19): "the pipeline is run once a month on a laptop; which laptop depends on who has time" — no production server exists |
| **Status** | Open |
| **Location** | `monthly_run.sh` (the entire production trigger: a hand-run bash list); no scheduler, no retry, no freshness check anywhere in the platform |
| **Notes** | Production is a **rotating human ritual**: whoever has time runs `monthly_run.sh` on their own laptop, with their own env/credentials state. Consequences: run evidence is scattered across personal machines (this workstation holds Apr/May traces only); "did month X happen?" is unanswerable from any repo; a missed or failed month is invisible until a human notices downstream. The maintainer's stated goal is a **dedicated small production server modeled on the working datafactory box** (which already runs monthly by timer) — gated on this repo's cleanup. Exit criteria for closing: (a) scheduled execution on a dedicated host, (b) a dead-man's-switch freshness alarm ("month-X artifacts absent by day D ⇒ scream"), (c) the laptop ritual retired to backup after one both-run-and-compare month. The wiring-acceptance instrument for that host now exists: **`python -m tools.liveness`** (epic #238, was planned as `tools/preflight`) — its exit-code contract (0 healthy / 1 attention / 2 unreachable) is the dead-man's-switch primitive; what remains for this entry's exit is *scheduling* it (cron on the future host + an alarm on non-zero), which no code here does yet. See also C-97, C-100, C-03 (no integration in CI). **Measured 2026-08-04:** the predicted silent failure has occurred and is quantified — FAO's forecast stream is **145 days stale** while a complete, deliverable run has sat on the internal shelf since 27 July. The specific mechanism at the delivery boundary is now **C-121** (no age bound on the resolved run); the missing scheduler remains this entry's. |

---

### C-100 — Config referencing external reality is validated nowhere; misconfiguration is discovered only by failing live

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Any config value naming an external object (Appwrite collection/bucket IDs, store names, env var families, `.netrc` hosts) drifts from what actually exists — the next live run fails at that step (best case) or silently misbehaves (worst case) |
| **Source** | delivery-machine map + the 2026-06-26 live failure (postmortems) |
| **Status** | Mitigated (2026-07-19: `tools/liveness` shipped, epic #238; residual = wiring it into an actual run cadence) |
| **Location** | Demonstrated: `APPWRITE_PROD_FORECASTS_COLLECTION_ID='forecasts_metadata'` did not exist in live Appwrite → un_fao smoke run died at store lookup (`views_pipeline_ERROR.log`, postmortem). Same class: all ~13 `APPWRITE_*` vars × both delivery sides, `.netrc` entries, store run-names |
| **Notes** | There is no preflight anywhere that checks config-vs-reality before a run touches production surfaces; the system's first contact with a wrong drawer-label is the live failure itself. **Exit: `tools/preflight`** (maintainer-proposed, design agreed): one read-only command auditing every declared external surface — inputs (viewser, datafactory zarr incl. `last_valid_month_id` vs partitions) and outputs (both stores, partner buckets, wandb) — with OK/FAIL/SKIP-with-reason semantics (truthful degradation per the C-75 lesson), runnable identically on a laptop and on the future production host, where it doubles as the acceptance checklist. **Mitigated 2026-07-19 (epic #238, S1–S8):** the exit exists as **`tools/liveness`** — `python -m tools.liveness` audits all six declared surfaces read-only (public API, datafactory zarr vs partitions, Appwrite `production_forecasts` incl. the REAL collection IDs that this entry's incident lacked, FAO `unfao_bucket` per stream, wandb execution, gjoll VPN store) with exactly the agreed semantics: raw facts, truthful SKIPs, exit 0/1/2, crash containment. The phantom `forecasts_metadata` ID is encoded as `HISTORICAL_WRONG_COLLECTION_ID` with the real IDs beside it (`tools/liveness/appwrite_store.py`), so this incident class is now machine-checkable before any live run. **Residual (why not Resolved):** the instrument is hand-run — nothing schedules it before `monthly_run.sh` or on a heartbeat (that is C-99's exit), and config *files* are still not diffed against reality (the check observes reality directly rather than validating each env var). See also C-96 (the zarr-freshness row, automated in `tools/liveness/datafactory_input.py`), C-97, C-99. |

---

### C-101 — tools/liveness verdict-truthfulness gaps: the instrument built to end false alarms can itself false-alarm or crash

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Running the dashboard on a machine without `datafactory_query` (reports UNREACHABLE/exit 2 instead of a truthful skip); any wandb project returning a run with a malformed/absent `created_at` (uncaught ValueError crashes the standalone check with no report); adding a new verdict without registering it in `EXIT_CODE_BY_VERDICT` (runner prints two contradictory verdict blocks for one surface); any error value containing newlines (breaks the one-fact-per-line contract — already visible in live vpn_store output) |
| **Source** | falsify (2026-07-19, claim "tools.liveness is air and water tight" → FALSIFIED, 3 hard / 3 soft) |
| **Status** | Resolved (2026-07-19, same-day fix: `one_line` newline escape in report.py; SKIP_NO_PACKAGE in datafactory_input; `_judge` inside the per-ensemble try; verdict classified before print in all six `main()`s; roster-mirror tripwire shipped — all enforced by `tests/test_liveness_falsifications.py`) |
| **Location** | `tools/liveness/datafactory_input.py:101-110` (generic except swallows ImportError → UNREACHABLE, no SKIP_NO_PACKAGE unlike vpn_store); `tools/liveness/wandb_execution.py:124-131` (`_judge` outside the per-ensemble try); `tools/liveness/__main__.py:44-47` + every module `main()` (exit_code_for raises AFTER print → double block); `tools/liveness/report.py:43-45` (render_facts passes newlines through); enforcement tests: `tests/test_liveness_falsifications.py` (failing by design) |
| **Notes** | Tier 2 rationale: structural fragility with named realistic triggers in the very instrument whose purpose is verdict truthfulness — a false UNREACHABLE from the dashboard re-creates the "who is lying?" failure mode it was built to end (C-75 class), and a crash-instead-of-report hides a surface. Root pattern (falsify pattern analysis): S7 extracted the renderer and exit map but NOT the exception/skip classification, so truthful-skip semantics are re-implemented per module and drift (vpn_store correct, datafactory_input not); contracts asserted in docstrings (one-fact-per-line, verdict-map totality, roster mirror) have no enforcement. Roster-mirror tripwire (monthly_run.sh vs MONTHLY_ENSEMBLES) ships with the fix. See also C-75 (truthful-skip lesson), C-94/C-95 (silent-when-unenforced class), C-100 (the incident class the suite mitigates). |

---

### C-102 — tools/liveness coverage gap vs its charter: viewser input, website host, and content-size judgment absent

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | **(a) RESOLVED 2026-08-24 (#411, S4 of #412) —** the surface now judges the ADR-013 commit marker by SUFFIX (`__manifest.json`), not a producer-specific filename; `unfao_delivery` reports `DELIVERING` against the live bucket where it had reported `NEVER_DELIVERED` over 110 delivered files. Matching the suffix rather than `rusty_bucket_forecasting_` keeps the surface out of the business of knowing which model produced the forecast. Guarded by a fake that DECODES the module's query and applies it to real 2026-08-24 file names, replacing one that keyed on the module's own constants and therefore asserted the tool agreed with itself — restoring the original matcher now fails that test loudly. `other_files` counts the manifest's whole run, so a healthy delivery no longer reads as 109 unattributed files. **~~(a) CONCRETE BUG, fires every run today —~~** `python -m tools.liveness unfao_delivery` judges the forecast stream on the *legacy* `forecast_dataset_*.parquet` name; the live path is now the ADR-013 manifest, so this surface reports `STALLED` on every healthy delivery **permanently** until it is taught the manifest. One-line-scope fix; do not leave it bundled with (b). **(b) SCOPE DECISIONS, need a maintainer call —** a viewser outage, a viewsforecasting.org website failure, or a truncated-but-present delivered file occurs while the dashboard reports all-green, because no surface watches viewser, no probe covers the website, and `*_newest_bytes` is reported but never judged |
| **Source** | falsify (2026-07-19, Category H adequacy probe against epic #238's charter "all input and output destinations") |
| **Status** | Open |
| **Location** | `tools/liveness/__main__.py` SURFACES registry: no viewser surface (the ACTUAL input of the four production ensembles; the suite watches the datafactory input production does not yet consume), no website probe (only `api.viewsforecasting.org`); `tools/liveness/unfao_delivery.py` reports `*_newest_bytes` but never judges them (a 12-byte parquet counts as DELIVERING) |
| **Notes** | **Half of this is resolved and the entry stays Open for the other half.** (a) — the wrong-name bug — was fixed 2026-08-24 (#411, S4 of #412); see the Trigger row. (b) — the scope decisions about a viewser surface, a website probe, and content-size judgment — is untouched and still needs a maintainer call, which is why Status remains `Open`. Tier 3 rationale: no wrong output is produced — the gap is scope, and the register + README non-goals note make it visible rather than silent. Closing requires a maintainer scope decision: (a) a viewser liveness surface (an S9), (b) a minimum-bytes/row-count judgment on delivered files (liveness vs content-sanity boundary), (c) whether the website is a distinct surface from the API or out of scope. Until decided, the README documents these as known non-goals so all-green cannot be over-read. **Run-0 note (2026-07-20, seat review):** two of these gaps become concrete at the first real FAO delivery — (1) `unfao_delivery` judges only file recency/bytes-present, not visible-to-consumer, so it read `DELIVERING` even while faoapi served nothing (the name-filter invisibility, C-97/C-100 axis); and (2) the un-judged `*_newest_bytes` means a truncated/empty run-0 upload would still read healthy. Neither blocks run 0, but the dashboard's green must not be over-read as "FAO can GET it" until faoapi's consumer wire (#100 S2) exists. **Run-0 confirmed, both directions (2026-07-27/28):** the gap fired for real, twice, in opposite directions. (1) **False red** — `unfao_delivery` judges the *legacy* artifact name (`forecast_dataset_*.parquet`), not the ADR-013 **manifest**, so it reported the forecast stream `STALLED` while a complete contract delivery (108 shards + sidecar + manifest) sat in `unfao_bucket`; now that the wire is the live path, this surface will read stale **permanently** until it is taught the manifest. (2) **False green** — the same surface read `DELIVERING` on that clean bucket while the API served the previous month's `orange_ensemble` (C-111). The instrument is watching the wrong artifact *and* the wrong hop. Exit for (1) is small and concrete: judge freshness on the manifest object, not the legacy parquet name; exit for (2) is the serving-truth surface described in C-111. See also C-99 (heartbeat), C-96 (input freshness class), C-97 (delivery identity), **C-111** (the serving-hop root this fails to observe). |

---

### C-103 — liveness test-suite taxonomy contradicts ADR-005; coverage below the "fully covered" bar

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Anyone running `pytest -m red` expecting the adversarial suite (gets six live network probes instead); `pytest -m beige` expecting structural compliance of tools/liveness (gets nothing); or trusting "fully covered" while refactoring the default network clients (95% branch coverage — precisely the default clients' error paths are unwatched) |
| **Source** | falsify (2026-07-19, claim "100% covered, green/beige/red all around" → FALSIFIED, 3 hard / 2 soft) |
| **Status** | Resolved (2026-07-19, same-day fix: ADR-005 amended with the `live` category + pyproject marker; 6 live probes relabeled; 22 error-path tests red-marked; 8 beige structural tests added; coverage closed to 100% branch — real tests for the default clients via fake modules, pragma only on `__main__` guards; taxonomy enforced by `tests/test_liveness_taxonomy.py`) |
| **Location** | All `tests/test_liveness_*.py`: live network probes marked `red` (ADR-005 red = adversarial/error-path, `pyproject.toml:3`); genuine error-path tests sit under file-level `green`; zero `beige` tests; measured 95% branch coverage (misses: default clients' error branches, `resolve_credentials` fallbacks, `__main__` guards). Enforcement stubs: `tests/test_liveness_taxonomy.py` |
| **Notes** | Root cause: marker *names* read from pyproject at S1 but not their *definitions* — "red = live/dangerous" was pattern-matched from a template test and WET-propagated through all six suites. Maintainer decision (2026-07-19): **Option A** — amend ADR-005 with a fourth `live` marker for tests that touch real external services; relabel the six probes `live`; red goes to the actual error-path tests; add beige structural suites; close the coverage gap (real tests for the coverable seams, `# pragma: no cover` only for `__main__` guards, with reasons). See also C-101 (same audit series), C-03 (tests not in CI). |

---

### C-104 — Posterior sample-count is four different config keys across model families, and the CI contract validates a key the runtime may not read

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A baseline (or stepshifter/r2darts) model's sample count is changed by editing `n_posterior_samples` (the key the CI contract and ensemble-parity test read) — but the model's forecast runtime reads a **different** key (`n_samples` / `pred_samples` / `num_samples`), so the change is silently ignored; CI stays green while the produced sample count is whatever the runtime key says |
| **Source** | investigation + falsify (2026-07-20, rusty_bucket FAO delivery: the "thinning" edited the decoy key and never changed the forecast) |
| **Status** | Open |
| **Location** | Runtime readers, one name per family: baseline `config["n_samples"]` (views-baseline `model/catalog.py:81`); hydranet `config["n_posterior_samples"]` (views-hydranet `utils/hydranet_inference.py:521`); r2darts `config["num_samples"]` (views-r2darts2 `engines/darts_forecasting_model_manager.py:579`); stepshifter `config["pred_samples"]` (views-stepshifter `models/shurf_model.py:24`). CI/parity contract reads **only** `n_posterior_samples`: `views-models/tests/conftest.py:152` (`get_n_posterior_samples`), consumed by `test_ensemble_configs.py` and `test_sample_count_standard.py` (ADR-015). No canonical definition or validator exists in pipeline-core. |
| **Notes** | Four model families name the same concept — posterior draws per cell — with four different config keys, while the runtime object and the ADR-013 wire already agree on one canonical name (`PredictionFrame.sample_count` / header `sample_count`). Only the **config layer** is fragmented. This became a **silent decoy** for baseline: C-52's readiness "resolution" (2026-06-02) *added* `n_posterior_samples` alongside the pre-existing `n_samples` on the older-convention models "derived from each model's existing `n_samples`" — but nothing keeps the two in sync, and the baseline forecast still reads `n_samples`. So a baseline config can carry `n_posterior_samples: 16` (CI-blessed) while forecasting at `n_samples: 128` (what actually runs), and every CI check passes. **Demonstrated 2026-07-20:** thinning rusty_bucket's constituents by editing `n_posterior_samples` had zero runtime effect (the forecast reads `n_samples`), costing four failed ensemble runs before the split was found. This is the **"knowledge" leg** of why C-85's stale-cache trap was so silent: even a correct operator edits a key that does nothing, with no signal. **The silence is three simultaneous failures on one axis** — *guardrail*: no cross-check that declared == runtime sample count (C-85); *knowledge*: four names + a decoy; *test*: the CI contract validates config-time on the decoy key, never the runtime-produced `pf.sample_count`. Exit (maintainer-directed, deferred — start with views-baseline + views-hydranet): one canonical key (`sample_count`, matching the frame/wire) read identically by all families; the config-time check reads the same key the runtime reads; and the authoritative check validates the produced `pf.sample_count`, which cannot drift from reality. Cross-refs: **C-85 (the stale-cache trap this made invisible)**, C-52 (the readiness resolution that seeded the decoy), C-74/C-116 (sample-count parity), ADR-013 §2 (`sample_count` canonical wire name), ADR-015 (the 128 standard, currently keyed on the decoy). **Cluster G** (added 2026-10-26, #610) — the CI contract validates a key the runtime may not read, so the gate's input is the wrong key and CI stays green while the produced sample count is whatever the runtime key says. |

---

### C-105 — The shared working checkout drifts durably behind `origin/development`; runs use stale code/configs

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A developer runs a model/ensemble, greps for a symbol, or inspects a config in the primary working checkout after recent merges — and gets stale results because the checkout was never pulled: all merges land via short-lived worktrees off `origin/development`, and the shared checkout is left behind |
| **Source** | repo-assimilation (2026-07-20) |
| **Status** | Open |
| **Location** | the primary working checkout at `~/Documents/scripts/views_platform/views-models` (observed 2026-07-20: HEAD `e266ec48` #237 vs `origin/development` `0ab28b47` #268 — **42 commits behind**; `tools/liveness/` absent locally though present on dev) |
| **Notes** | The worktree-based merge workflow (adopted to protect parallel sessions sharing the tree) durably decouples the primary checkout from `origin/development` — it is only ever *branched from*, never *pulled into*. Consequences: a run from the stale checkout uses old code/configs (e.g. would miss the S1/S2 sample-count guards); a grep for recently-merged files (`tools/liveness`) finds nothing though they exist on dev; and the drift interacts sharply with the **editable-install model** (the `views_pipeline` env runs sibling repos in `-e` mode, so *local checkout state IS the running code* — confirmed for views-reporting 2026-07-20). Not corruption and no wrong output on its own, but a real "what is actually running / present here?" hazard that compounds every debugging session. Mitigation direction (not applied): a periodic `git pull` / fast-forward discipline on the shared checkout, or a convention that the primary checkout tracks `origin/development`. See also C-42/C-50 (the fresh-clone/editable-install *dependency* class — related but distinct: those are about a clean env failing to resolve deps, this is about an existing checkout being out of date), and **C-110** (the inverse direction of the same hazard: run-critical config living *ahead* of the remote, uncommitted and unbacked). **Update 2026-07-28:** the primary checkout was fast-forwarded to `origin/development` and its branch renamed to `development` (ff-only, triple backup tags) after being found parked on a merged feature branch 7 commits behind — the drift is real and recurring, and the mitigation direction above is still unapplied as a *discipline*. Member of **Cluster H** (no identity separates current from left-over). |

---

### C-106 — STRATEGIC ROOT: the test architecture verifies declarations exhaustively but runtime behavior nowhere in CI — the config-vs-behavior gap

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A developer merges a change that is **declaration-valid but runtime-wrong** — an edited hyperparameter with wrong *semantics* (not shape), a key the runtime doesn't read, a manager-call change, a queryset/return-shape drift — the full config/structure suite passes green, the change merges, and the defect surfaces only at a manual `run_integration_tests.sh`, a hand-run ensemble, or in production |
| **Source** | repo-assimilation + test-review (2026-07-20, "strategic") — synthesis of a recurring pattern already scattered across the register |
| **Status** | Mitigated (2026-07-20, PR #272): the stated exit exists — `tests/test_runtime_smoke.py` executes 21 offline distributional-baseline models through the real config→catalog→model path (asserting sample_count==config [C-104], shape, dtype, no NaN/Inf, non-negative, determinism) in ~0.3s, and `.github/workflows/runtime_smoke.yml` runs it as a dedicated green+required PR check that dodges the published-core skew (installs no pipeline-core). One end-to-end runtime path is now verified at PR. **Residual (why not Resolved):** coverage is the baseline family only (hydranet/r2darts/stepshifter still declaration-only); it drives the catalog seam, not the full main.py→manager path; the main suite remains red (C-80) so the smoke's green signal lives in a separate job; **and the mitigation was itself blind for five weeks — see Notes, 2026-09-09.** |
| **Location** | The test *architecture*: `tests/` (parse-based, `importlib`/AST, parametrized over `ALL_MODEL_DIRS` in `conftest.py`) verifies config validity, structure, and declared contracts exhaustively (~7100 passing) — while **runtime/production behavior is verified nowhere in CI**: `run_integration_tests.sh` is the only runtime check and is manual by its own CIC's non-goal (C-03); ADR-005 accepts source-based tests cannot validate runtime |
| **Notes** | This is the **causal root** that a large cluster of existing entries are individual instances of, none of which names the pattern itself. The test strategy is deliberately declaration-oriented (fast, no-ML-deps, green in crippled CI — ADR-005) and is genuinely excellent *at that layer*. But it draws a hard line: **"the config is valid" is proven thousands of ways; "the system actually works" is proven zero ways in CI.** Every recurring silent-failure incident lives in that gap — the runtime reads a key CI never checks (C-85, C-104), a datafactory aggregation the config can't express (C-94), a feature-map that rots unseen (C-95), a `generate()` return-shape crash (C-40, C-02) — and the structural gaps that enable them are C-03 (integration not in CI), C-16 (CIC guarantees output-tested not behavior-tested), C-32/C-33 (test-passes-but-runtime-crashes), C-80 (no green baseline so even a runtime regression that *did* surface would be lost in noise). **Strategic exit (the highest-leverage single lever in the repo):** one minimal *runtime* smoke in CI — even a single tiny synthetic model trained+forecast on a 2-cell fixture — converts an entire class of currently-invisible failures into caught-at-PR failures. It does not need the full fleet or heavy deps; it needs *one* end-to-end execution path that CI actually runs. Until that exists, config-green will keep meaning "declared correctly," never "works," and the silent-failure incidents will keep recurring one config key at a time. Cross-refs (the cluster this anchors): C-03, C-16, C-02, C-40, C-32/C-33, C-80 (mechanisms/gaps); C-85, C-94, C-95, C-104 (incidents); ADR-005 (the accepted source-based-testing limitation this makes strategic). **2026-09-09 (#460) — the mitigation was green for the wrong reason, and this is the entry's sharpest lesson.** `runtime_smoke.yml` installed views-baseline from a hardcoded git ref, `b53fc41`, `--no-deps`. That commit is 2026-07-20, **35 commits behind v1.0.2**, and its `catalog.py` still read the `targets` key at 7 sites. views-pipeline-core retired that key in `507ae11` on 2026-08-02, which made **all 37 baseline models unrunnable** (#445). The runtime smoke — the one job whose entire purpose is to answer *does a model actually run* — **stayed green for the whole five weeks**, because it was pinned to code no model could run. It was not measuring the fleet; it was measuring a frozen snapshot of a sibling repo. Two guards failed together: the workflow pin hid the break from CI, and `tests/test_runtime_smoke.py` hand-built the retired key (`config = {**hp, "targets": targets}`) so the test passed on 1.0.1 **and** 1.0.2 — it compensated for the defect it existed to catch. #460 fixes both (PyPI release pins; the injected key replaced by the live `regression_targets`, verified red on 1.0.1 and green on 1.0.2). **The generalisable rule: a runtime guard pinned to a frozen upstream ref stops being a runtime guard the moment the upstream moves — it silently becomes a regression test for a snapshot.** Any CI job that exists to answer *does this still work against our dependencies* must track a released version, not a commit. Cross-refs: **C-116/C-117** (environment contents are undeclared, the same class), C-113 (a gate that cannot tell "clean" from "did not run"), #204 (satisfaction-not-currency in `run.sh`, the sibling mechanism that kept stale envs stale). |

---

### C-107 — `tools/liveness` (a substantial new subsystem) has no CIC and no governing ADR

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | A developer modifies a `tools/liveness` surface-check class's contract — a verdict enum value, the exit-code map, the injected fetch/clock seam, or the truthful-skip semantics — with no CIC to check the change against: the only specification is the code plus the README, so a silent contract change (e.g. a verdict that no longer maps to its documented exit code) has no governance tripwire |
| **Source** | review-base-docs (2026-07-20) |
| **Status** | Mitigated (CIC authored, 2026-07-21, #275) |
| **Location** | `tools/liveness/` — 8 modules / 6 surface-check classes (`old_api.py`, `datafactory_input.py`, `appwrite_store.py`, `unfao_delivery.py`, `wandb_execution.py`, `vpn_store.py`) sharing a `run() -> verdict` + injected-fetch + exit-code contract; 130 tests; epic #238. Now governed by `docs/CICs/LivenessChecks.md` |
| **Notes** | Under **ADR-006** (intent contracts for non-trivial classes), the six liveness check classes — each a cohesive contract with an injected-dependency seam, a verdict enum, and an exit-code mapping — warrant a CIC, and the suite as a whole (an operational instrument used platform-wide) arguably warrants an ADR. It had neither. **Well-mitigated**, which is why Tier 4 not 3: a thorough `tools/liveness/README.md` documents every surface, verdict, exit code, and the encoded conventions with receipts; and ADR-005 (§ the `live` category) + ADR-017 (§ the observability instrument for derived state) both reference it. So the *contract existed in prose* — the gap was that it was not in the machine-checkable CIC form the repo's own convention prescribes, so `validate_docs.sh` and the CIC-audit tools could not guard it. **Resolution (#275, 2026-07-21):** authored `docs/CICs/LivenessChecks.md` — one subsystem-level CIC (the `ReconciliationWiring.md` precedent) covering the shared check-class contract, referencing the README as the human companion; wired `tools/liveness/report.py` and `tools/liveness/__main__.py` into `.github/workflows/cic_sync_check.yml` so a change to the verdict/exit-code map or the runner now forces the CIC to move in the same PR. A dedicated **ADR** for the suite remains optional (ADR-017 already leans on it) — deliberately not built. Cross-refs: C-103 (same subsystem, the test-taxonomy aspect — resolved), C-16 (CIC coverage of non-trivial classes), C-106 (the runtime-smoke work, similarly under-governed — no ADR yet). |

---

### C-108 — No durable, access-controlled source of truth for platform secrets

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A laptop holding the only copy of the production Appwrite datastore API key dies / is reimaged / the colleague leaves; or a fresh checkout must run a production step. Today there is no durable backup to restore from and no documented provisioning — recovery means scrambling to regenerate keys in the Appwrite console (a rotation), and every consumer must be re-provisioned by hand. |
| **Source** | credential audit 2026-07-27 (`reports/security/appwrite_credentials_audit.md`) |
| **Status** | **Subsumed by þing-01 (The Appwrite Seam Contract, then named PLATFORM-001), 2026-07-28** — the secrets-management architecture is decided by the ratified cross-repo verdict; the risk is now owned by that contract + its per-repo follow-through, not by this entry (was: Open, architecture deferred). |
| **Location** | Platform-wide. Producer/postprocessor read via `load_dotenv` → `views-models/.env` (pipeline-core `configs/prediction_store.py` `_ENV_MAP`; postprocessing `unfao/managers/unfao.py`). Server: `views-faoapi/deployment/bootstrap.sh:21` hardcodes `SOURCE_ENV=/home/sonja/.../views-models/.env`. |
| **Notes** | The 15 Appwrite keys (+ `ACLED_PASSWORD`, `GDL_API_TOKEN`, `UCDP_API_TOKEN`, `VIEWS_DATAFACTORY`) that a UN-facing production service depends on exist **only** in personal, gitignored `.env` files on ≤2 laptops plus one derived server copy (`.env.faoapi`). **No durable, backed-up, access-controlled source of truth; no independent per-person revocation; no rotation runbook; the deploy bootstrap is pinned to one individual's home directory.** This is the root cause of credentials repeatedly being re-supplied across sessions ("the mystery"). *Not a leak:* the audit verified **no secret is in git, in any repo, across full history**, and none is pasted into any tracked file — so this is a **fragility / recoverability** risk, not exposure. **Vocabulary is consistent** (one canonical naming across all repos); the problem is *provisioning architecture*, not sprawl. **Immediate hygiene shipped (this entry's partial):** `views-models/.env.example` (the documented schema), `tools/check_credentials.py` (self-diagnosing "which keys am I missing?"), and `tests/test_credentials_presence.py`. **Deferred by maintainer decision (2026-07-27), informed by an external assessment:** the real secrets-management architecture is a deliberate investigation, NOT chosen now — options are SOPS+age with individual keys (interim/low-complexity, per-person revocation), a secrets manager / OIDC short-lived creds (production automation), or a team password manager (human-accessed); GPG-shared-passphrase-in-a-repo is explicitly *interim bootstrap only, not the target*. Tracked in views-models#280. Cross-refs: the monthly-run-ritual entry (same "scattered on personal laptops" family), C-79 (infra/IP-in-config hygiene). **Subsumed (2026-07-28):** the deferred investigation (views-models#280) converged at þing-01 into **The Appwrite Seam Contract** (named `PLATFORM-001` at the time) — one identity/secrets/config contract homed in views-appwrite, an owned coordinate registry, a declared secret/coordinate split, three credential tiers, and raise-by-default failure semantics. The risk persists until the contract is fully implemented, but it is now owned by that contract, not this entry. views-models follow-through: #285 (this retarget), #286 (harvest-token deletion), #287 (launcher registry read), #288 (curation-list re-homing). Contract: https://github.com/views-platform/views-appwrite/blob/856d617/docs/ADRs/platform/appwrite_seam_contract.md — **v1.3.0**, pinned at `856d617`. (This citation read `60674b2` until 2026-08-02, which is **v1.0.0** — two ratified versions behind (v1.2.0, v1.3.0; v1.1.0 was proposed and never ratified); nothing signalled that, which is the argument ADR-011 makes for readable identifiers over opaque ones.) |

---

### C-109 — Unpaginated Appwrite `list_documents` silently truncates at 25; "the whole set" is a 25-item lie

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Any code that enumerates Appwrite documents to build a *complete* set — resolving a wire manifest's shard list, listing every delivered file for a month, counting artifacts to decide "is the run finished?" — and calls `databases.list_documents(...)` (or the REST equivalent) without an explicit `Query.limit()` + offset loop. The set silently caps at the SDK default of 25 and the caller treats the truncated slice as the whole. |
| **Source** | run-0 serving incident (2026-07-28), root-caused live during the FAO delivery |
| **Status** | Open (the *incident* is fixed in views-faoapi; the *pattern* is unguarded platform-wide) |
| **Location** | Incident: views-faoapi `src/views_faoapi/managers/appwrite.py:838-870` `search_files_by_metadata` — `list_documents(db_id, coll_id, queries=queries)` with no limit/offset → resolved **25 of 108** shards for the run-0 manifest → ingest refused → the API served the previous month's `orange_ensemble` for ~24h while a clean run-0 sat in `unfao_bucket`. Fixed in views-faoapi#287 (offset loop, `DEFAULT_PAGE_LIMIT=100`), deployed v1.3.2. views-models exposure: `tools/liveness/appwrite_api.py:90` `newest_first_query(limit=…)` — always passes an explicit limit and only ever asks "newest N", so it does not currently manifest, but nothing enforces that. |
| **Notes** | Tier 2 rationale: structural fragility with a named, already-realized trigger — a client-library **default** silently redefines "all" as "the first 25", and the caller has no signal. It cost the first-ever UN-facing global-land delivery a day of serving a stale forecast. What converted silent-partial into loud-refuse in this instance was luck of design: the ADR-013 manifest **declares its shard count**, so the consumer could compare 25 against 108 and refuse. Any listing that is *not* cross-checked against a declared expected count would have produced a **silently partial dataset** instead — the same defect one hop away from Tier 1. Exits (either or both): (a) a shared paginated-list helper as the only sanctioned way to enumerate Appwrite documents in every repo, so the default is never reachable; (b) the ADR-013 discipline generalized — every enumeration that must be complete carries a declared expected count and refuses on mismatch. Cross-refs: C-97 (the addressing/identity axis of the same interchange), C-100 (config-vs-reality discovered only by failing live), C-111 (the fallback that hid this failure from the producer). **Cluster G** (added 2026-10-26, #610) — the listing silently caps at the SDK default of 25 and the caller treats the truncated slice as the whole set. C-169's `-xdev` in a different API: the search domain is set by a default rather than by the question. |

---

### C-110 — The configuration that produced the live UN-facing forecast exists only in an uncommitted working tree

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Anyone re-runs `postprocessors/un_fao` from a clean checkout, a fresh clone, or after a `git checkout -- .` / stash-drop in this tree — the committed configs still say `REGION = "africa_me_legacy"` (13,110 cells, pandas path), so the re-run **succeeds** and delivers a Middle-East+Africa forecast into `unfao_bucket` under the same product name as the global-land one now being served. No error, no warning, wrong extent. |
| **Source** | run-0 delivery (2026-07-27/28); confirmed still uncommitted 2026-07-31 (`git status`) |
| **Status** | Open |
| **Location** | `postprocessors/un_fao/configs/config_queryset.py` (`REGION` `africa_me_legacy`→`land_gaul`; `"data_format": "feature_frame"`; datafactory pin `>=1.9.0,<2.0.0`), `postprocessors/un_fao/configs/config_meta.py` (`wire_contract`, `wire_upload_enabled`, `region: land_gaul`), `postprocessors/un_fao/README.md`, `postprocessors/un_fao/requirements.txt` — all `M` in the working tree, none on `origin/development` |
| **Notes** | Tier 2 rationale: not silent corruption *today* (the served artifact is correct and its provenance is verifiable on the shelf), but a structurally fragile state with a realistic, one-command trigger and a wrong-output consequence — the delivered product is **not reproducible from any committed state**, and the reproduction attempt fails *quietly and plausibly* rather than loudly. The working tree is acting as the system of record for a UN-facing deliverable. Aggravating factor: the same tree also holds parallel-session-owned `violet_visitor` edits, so it cannot simply be committed wholesale — the un_fao paths must be staged by name. **Exit: commit the four un_fao paths via the merge ritual** (small, ready, blocked on nothing). Related cross-repo state, unverified from this repo and therefore not registered separately: the views-postprocessing bug-#1 name-scoping fix (`_prod_forecasts_datastore(name_scoped=False)`) was reported uncommitted in that checkout and may have the same exposure — worth a check in that repo. Cross-refs: **C-105** (the *inverse* direction of the same "working tree is the system of record" hazard — that entry is the checkout drifting *behind* the remote, this one is run-critical state living *ahead* of it and unbacked); C-53 (config value regression across merges). **Second variable found 2026-08-04, same file, same shape:** `postprocessors/un_fao/configs/config_meta.py` carries `wire_upload_enabled: True` in the working tree and **nowhere in git**. That key is the ADR-013 §11.4 upload interlock — with it absent the sink stages locally and makes ZERO store calls; with it present the run publishes to the UN FAO's bucket. So **whether this platform delivers to a partner is decided by an uncommitted edit on one laptop**, and two identical checkouts behave differently toward an external party. Registered here rather than as a new entry because it is the same defect at the same location: the configuration governing a UN-facing delivery exists only in a working tree. **Exit:** commit the key with whatever value is intended, so the deployed behaviour is derivable from a commit. Cross-ref: **C-117** (the dependency half, mitigated). |

---

### C-111 — The serving hop degrades silently: a failed ingest falls back to last-good, and the producer sees nothing

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A delivered run is malformed, partial, or hits any consumer-side ingest bug — `views-faoapi` logs the failure server-side, keeps serving the previous month's forecast as if current, and **no signal reaches the producing side**: not the postprocessor (already exited 0), not `tools/liveness` (which judges bucket recency, not what the API actually serves), not a human. The stale forecast is discovered only when someone thinks to ask. |
| **Source** | run-0 serving incident (2026-07-27/28) — Hop 2 delivered clean, faoapi served `orange_ensemble` (March) for ~24h, discovered only by direct maintainer question |
| **Status** | Open |
| **Location** | views-faoapi `src/views_faoapi/managers/dataset_service.py:621-720` `_load_wire_run` (lazy on-request ingest; refuse → fall back to last-good) — by design, and correct as a *availability* policy; the gap is that the degradation is invisible upstream. views-models side: `tools/liveness/unfao_delivery.py` observes the FAO bucket, not the served response; there is no surface for "what is the API serving right now, and is it the run we shipped?". |
| **Notes** | Tier 2 rationale: structural fragility across a repo boundary with a realized trigger and a stakeholder-visible consequence — UN consumers received a month-old forecast presented as current, with no error anywhere in the producer's view. Serve-last-good is the *right* availability choice; the defect is that it is a **silent** downgrade, so "delivered" and "served" are two different truths with nothing comparing them. Compounding observability defects seen the same session: the API's `/version` endpoint lagged the deployed git tag by one release (reported `1.3.4` on `deployed_tag v1.3.5`), so "which build is live?" could not be answered from the API itself; and `/provenance/forecast`'s top-level fields carried the *previous* run's name/filename/`run_id` (views-faoapi#290, fixed v1.3.5) so even a correct serve looked wrong. Exits: (a) a **serving-truth liveness surface** in `tools/liveness` that reads the live API's provenance and compares `run_id` against the newest manifest in `unfao_bucket` — this is the check that would have caught run-0 in minutes and is squarely a views-models deliverable; (b) upstream, a degraded/stale flag in the served response or an ingest-failure signal the producer can poll. Cross-refs: **C-102** (the detection half — `unfao_delivery` reads `DELIVERING` while the API serves nothing; this entry is the *root* it fails to see), C-97 (delivery identity), C-99 (no heartbeat), C-109 (the ingest bug this fallback concealed). |

---

### C-112 — Presence checks read SHELL scope while consumers need EXPORTED scope; the same blind spot has now shipped twice in four days

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Any shell code that decides "is this credential/coordinate available?" with `[ -n "$VAR" ]` or `[ -n "${!NAME}" ]`, in a script that has previously `source`d a `.env`. Sourcing sets the variable in **shell** scope; the check passes; the child process — which is what actually needs it — receives nothing. |
| **Source** | code-review max on PR #315 (2026-08-02), reproduced directly; earlier instance found during þing-02 (`orð_09 §2H`) |
| **Status** | Open |
| **Location** | Instance 1 (2026-07-31, fixed in #314): `postprocessors/un_fao/run.sh` `_platform001_coordinate_state()` — announced *"Coordinates ARE present in the environment (exported outside this script)"* on the strength of an unexported shell variable. Instance 2 (2026-08-02, caught pre-merge in #315): `tools/credentials/platform_env.sh` `platform_env_export_secret()` guard `[ -n "${!PLATFORM_ENV_SECRET_NAME:-}" ]` — returns early because `run.sh` sourced `.env` for `GITHUB_TOKEN` two dozen lines above, so the `export` is never reached; `platform_env_validate()` shares the blind spot and reports the environment complete. |
| **Notes** | **The recurrence is the finding, not either instance.** Both were written by an author who had just read #293 — the incident whose entire content is *"`source` without `export` does not reach the child"* — and both reproduced it anyway, because bash makes the two scopes indistinguishable at the point of test. `[ -n "$VAR" ]` cannot tell them apart; only `[ -n "${VAR+x}" ]` combined with an `export -p` lookup, or a probe of an actual child process, can. **Reproduced on the PR branch:** with `.env` sourced first, `platform_env_export_secret` returns 0, `platform_env_validate` passes, and `python -c 'os.environ.get(...)'` returns `None` — a run that reports a complete environment and hands the child nothing. Instance 2 was additionally *protected by a test*: `test_the_launcher_does_not_export_the_secret_itself` asserts the launcher must not export the secret directly, which is correct as a one-writer rule and, combined with the defective guard, enforced the bug. **Exit:** a single sanctioned way to answer "will the child see this?" — probe the exported environment (`env` / a real child), never the shell's. Everything else in this class is a rediscovery. Tier 2: silent, produces a successful-looking run that delivers nothing, and the demonstrated recurrence rate is twice in four days. Cross-refs: **C-111** (silent serving degradation — same family: success reported, nothing delivered), C-94/C-95 (silent-when-unenforced), C-57 (the sibling class where a *regex* cannot distinguish a comment from code, which also recurred twice this week). Member of **Cluster G** (a check out of scope returns a clean verdict). **Moved from Cluster A during review-rr (2026-10-25)** — the defect here is a check whose verdict is uninformative, not a declaration nothing enforces. **A sibling shape, found in the same review and fixed rather than registered** (one instance, now pinned by a test): a guard written for the **steady state** was reused during **setup**, where the condition it treats as fatal is the normal starting state — `bootstrap.sh` called the fatal `platform_env_export_secret` before prompting, so a first-ever machine's very first output was *"FATAL: … does not exist. Run ./bootstrap.sh"* addressed to the person running `./bootstrap.sh`. Not the same defect as this entry, but the same question badly posed: **a check must be asked in the context it will answer for.** |

---

### C-113 — A feature branch was committed with no parent, and every gate in the merge ritual passed it

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Any `git commit` that is interrupted (timeout, Ctrl-C, a hung prompt) and then re-run. If the interruption left `HEAD` unborn, the retry produces a **root commit** and the branch silently becomes an orphan with no common ancestor. Every subsequent commit extends the orphan. |
| **Source** | Pre-merge safety gate on PR #315 (2026-08-02) — caught by hand, not by any check |
| **Status** | Mitigated |
| **Location** | `feat/309-311-platform-env-and-bootstrap`, commit `0651448f` (`parents: []`). Five commits, 5 reachable ancestors against `development`'s 877. Repaired by re-parenting each tree with `git commit-tree` onto `2554d731` and moving the branch with `git reset --soft` (no working-tree write — the maintainer had six dirty files including two in `models/violet_visitor/`, which was running experiments). |
| **Notes** | **What made it dangerous is that nothing reported it.** `git status`, `git log`, `ruff`, the 7,268-test committed-state suite, four rounds of code review, and `gh pr view` all passed: the *tree* was correct (development's content plus exactly nine intended files), only the *history* was disjoint. GitHub reported `mergeable=MERGEABLE` and offered the merge button. The only signal was `git diff development...branch` failing with `fatal: no merge base`, rc=128 — and on the first attempt **that failure was itself swallowed**: the command's error text went to a variable, the empty result was grepped for `violet_visitor`, found nothing, and the gate printed *"violet_visitor: NOT touched ✓"*. A gate that cannot distinguish "clean" from "did not run" is not a gate. **Probable cause:** a `git commit -m` whose message contained backticked shell (`` `read -rs` ``) — bash executed `read`, which blocked on stdin for two minutes until killed. **Exits, in order of value:** (1) every safety gate must check the exit status of the command it reasons about and refuse to report a verdict on empty output; (2) always `git commit -F <file>`, never `-m` with a message containing backticks; (3) a cheap pre-merge assertion — `git merge-base --is-ancestor origin/development HEAD` — would have caught this in one line. Tier 2: silent, survived every existing control, and the failure mode is a merge of disjoint history into `development`. Cross-refs: C-112 and C-57 (same family — a check that cannot distinguish the state it claims to test), **C-94/C-95** (silent-when-unenforced). Member of **Cluster G** (a check out of scope returns a clean verdict). **Moved from Cluster A during review-rr (2026-10-25)** — the defect here is a check whose verdict is uninformative, not a declaration nothing enforces. **Recurrence 2026-08-04 (#343):** the same reading — *empty output means clean* — appeared again in a new test, `tests/test_deliveries_characterisation.py::test_nothing_reads_deliveries_yet`, which shells out to `git grep` and treated empty stdout as "nothing references `deliveries/`". `git grep` exits **128 with empty stdout** when it cannot run (wrong cwd, bad pathspec), so the guard would have passed on error — in the one test whose whole job is proving that story changed no behaviour. Caught in that PR's own code-review step and fixed by asserting `returncode in (0, 1)` before reading stdout. **Why this is recorded rather than shrugged off:** the lesson recurred in a different subsystem two days after being written down, which is evidence it does not transfer by being documented. Any test that shells out must check the exit code before interpreting output. |

---

### C-114 — The detector built for the November key expiry reported a rejected key as mild staleness

| Field | Value |
|---|---|
| **Tier** | 1 |
| **Trigger** | The Appwrite datastore key expiring, being revoked, or being replaced with a wrong value — expected around **2026-11-30** for both current keys. Also any code that judges Appwrite reachability from a **file-listing** response alone. |
| **Source** | Building the #302 preflight (2026-08-02); found because the preflight was tested against a simulated dead key before being trusted |
| **Status** | Resolved |
| **Location** | `tools/liveness/appwrite_store.py` and `tools/liveness/unfao_delivery.py` — both read only `GET /storage/buckets/{id}/files`. Fixed by `assert_bucket_reachable` in `tools/liveness/appwrite_api.py`, called first inside each surface's existing `try`. |
| **Notes** | **Appwrite answers the file-listing endpoint with HTTP 200 and `total: 0` when the key is rejected.** Measured three ways against the live server (Appwrite 1.9.5, 2026-08-02): real key → 200, `total=461`; garbage key → 200, `total=0`; empty key → 200, `total=0`. Listing files was the *only* call either surface made, so a dead credential was indistinguishable from an empty bucket. `appwrite_store` returned `STORE_IDLE` with `error: bucket contains no files`; `unfao_delivery` would have returned `DELIVERY_STALLED`. Both are **exit 1, "attention"** — a verdict a human reads as "nothing landed lately", which is unremarkable for a monthly cadence. **Why Tier 1 rather than 2.** This is not a check that might mislead in principle; it is the check this platform designated as the detector for a *known, dated* silent failure. C-99 records that the write path logs *"Forecasts uploaded successfully"* while uploading nothing once the key dies; #302 exists to schedule this detector against that date; and the detector renders exactly that failure as ordinary staleness. Both the alarm and the thing it watches were silent in the same way, so the platform would have concluded "quiet month" through a full delivery cycle to an external partner. **The fix, and why the bucket GET.** Every other endpoint tested returns 401 for the same rejected key — bucket get, bucket list, database get, collection list, `/health`. `assert_bucket_reachable` GETs the **bucket itself** because it settles two questions in one call: the key is accepted (401 if not) and the bucket coordinate still resolves (404 if not) — a wrong bucket id would otherwise also have surfaced as emptiness, for the same reason. Verified live afterwards: real key → `STORE_ACTIVE`, 461 files; garbage key → `UNREACHABLE`, `HTTP Error 401`, exit 2. **What generalises.** *An empty result and a refused request are the same bytes unless something distinguishes them.* Cross-refs: **C-99** (the alarm is built and nothing runs it — this entry is why scheduling it was not yet sufficient), **C-111** (silent serving degradation), **C-112** (a check asked in a scope that cannot answer it), **C-113** (a gate that cannot tell "clean" from "did not run"). Member of **Cluster A** (declared-but-unenforced). Contract updated in `docs/CICs/LivenessChecks.md` §6. **Cluster G** (moved from Cluster A, 2026-10-26, #610) — something *was* checked; the detector's input simply could not see rejection, so it returned a benign verdict. That is a check out of scope, not a declaration nothing enforces. |

---

### C-115 — A version boundary is encoded as a hyphen: merging two look-alike env directories silently gives 31 models the wrong package

| Field | Value |
|---|---|
| **Tier** | 1 |
| **Trigger** | Anyone normalising the inconsistent `env_path` names (`envs/views_r2darts2` vs `envs/views-r2darts2`, `envs/views_stepshifter` vs `envs/views-stepshifter`), or a `run.sh` regenerated from views-pipeline-core with a normalised path. It reads as fixing a typo. |
| **Source** | expert-code-review (2026-08-02), env x declared-spec cross-tab |
| **Status** | Open |
| **Location** | `envs/views_r2darts2` (22 tenants) vs `envs/views-r2darts2` (9 tenants); the 31 r2darts models' `requirements.txt`; `env_path=` line 18 of each `run.sh` |
| **Notes** | **The two directory names differ by one character and that character is load-bearing.** `envs/views_r2darts2` holds 12 models declaring `views-r2darts2==0.1.0` and 10 declaring `>=0.1.0`; `envs/views-r2darts2` holds 9 declaring `>=1.0.0,<2.0.0`. `==0.1.0` and `>=1.0.0,<2.0.0` are **mutually unsatisfiable**. They do not collide today only because they resolve to separately-named directories. Merge the names — the obvious tidy-up — and `run.sh` installs each tenant's own file into one shared prefix with no uninstall, so the resolved version becomes whatever ran last. Half the models then run a version they did not ask for. **Tier 1, not 2:** there is no error. pip reports success for each individual install; the models train and emit forecasts; the forecasts are computed by the wrong algorithm version. The counterpart split `envs/views_stepshifter` (32) vs `envs/views-stepshifter` (7) declares **identical** specs and is pure duplication (~9G each on disk) — so the naming inconsistency is meaningful in one place and meaningless in the other, with nothing distinguishing them. **Exit:** rename to state the constraint (`envs/views_r2darts2_v0` / `_v1`) or pin the split with a test whose failure message explains it. A five-line test is the cheapest high-value change available. Cross-refs: **C-116** (the shared-environment root cause), **C-70** (`run.sh` duplication), C-39 (generator-sourced regression). Member of **Cluster A** (declared-but-unenforced). |

---

### C-116 — 131 requirements.txt resolve into 11 shared environments, so a model's dependencies are decided by its co-tenants' run order

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Two tenants of one environment needing different versions of the same package — or anyone reading a `requirements.txt` to answer "what will this model run with?" |
| **Source** | expert-code-review (2026-08-02); measured cross-tab of `env_path` against declared specs |
| **Status** | Open |
| **Location** | 131 `requirements.txt` -> 11 `envs/*` values. `envs/views-baseline` 37 tenants, `envs/views_stepshifter` 32, `envs/views_r2darts2` 22, `envs/views_ensemble` 13, `envs/views-r2darts2` 9, `envs/views-hydranet` 8, `envs/views-stepshifter` 7, plus 4 singletons. Install logic: `run.sh:22-39`. |
| **Notes** | **Per-model *declaration* is real; per-model *isolation* is not, and the repo reads as though both were.** Each `run.sh` pip-installs only its own `requirements.txt` into the shared prefix and never uninstalls, so environment contents depend on which tenant last ran. **Proven in both directions.** *Declared but absent:* `ensembles/skinny_love/requirements.txt` declared `views-frames>=1.7.0,<2.0.0` while `envs/views_ensemble` has views-frames not installed — and skinny_love completed a run in that state on 2026-07-22 (wandb `atomic-jazz-101`, `state=finished`). *Present but undeclared:* 27 models receive views-datafactory because a co-tenant declares it; in `envs/views-baseline` only 10 of 37 tenants declare it. To answer what a model will run with you need four facts — its own file, its `env_path`, its co-tenants' files, and the order they last ran — and three of them are not in the file you are reading. **Not a call to centralise:** the `config_partitions.py` precedent stands and the files must stay per-model. What is missing is that the *shared* thing has no declaration at all. **Exit (smallest honest):** (a) `pip freeze` per run stored with the artifact — see C-117; (b) a generated comment in each `requirements.txt` naming its environment and co-tenant count, which pulls the three hidden facts into the file being read. Cross-refs: **C-38** (a specific instance — `datafactory_query` absent from an env that must run bright_starship), **C-115** (the unsatisfiable case this enables), **C-70**, C-08. Member of **Cluster A**. |

---

### C-117 — The dependency closure that produces a UN-facing forecast exists only on one laptop's disk

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Anyone asking which package versions produced a specific delivered forecast — or a forecast being questioned by the partner. |
| **Source** | expert-code-review (2026-08-02, Nygard/Kleppmann) |
| **Status** | Mitigated |
| **Location** | `envs/` (gitignored); `run.sh` (no resolved-manifest capture); `monthly_run.sh` |
| **Notes** | Environments are gitignored, so the versions a run used are a property of one machine's disk and of no commit. Measured on the maintainer's laptop: **3 of the 11 environments exist at all**, totalling 22G (`envs/views-baseline` alone is 9.0G) — full provisioning would be roughly 100-200G, which is also why 11 shared environments rather than 131 is the correct resource decision and not laziness. The consequence is that the repo's own stated rule — *"if it is not committed to git, you cannot assume it exists"* — is violated by the dependency closure of a UN-facing deliverable. **This is the dependency twin of C-110**, which records the same failure for *configuration*; registered separately because the artifact, the owner and the fix differ, but they should be closed together. **Exit:** one line in `run.sh` — `pip freeze > logs/<run_id>_env.txt` — kept with the forecast artifact. This converts "which versions produced this?" from unanswerable to a lookup and is the highest-value change surfaced by this review, ahead of any hygiene work. Cross-refs: **C-110** (configuration provenance), **C-116** (why the environment is not derivable), C-10 (`envs/` in the tree, Accepted), C-97/C-98 (delivery identity and system of record). Member of **Cluster A**. **Mitigated 2026-08-03 (#327):** `monthly_run.sh` now writes a `pip freeze` per environment into `reports/env_snapshots/`, after each folder runs (before would describe the environment as it was, not as it was used), deduplicated so the four ensembles sharing `envs/views_ensemble` produce one file. The header carries the run id, the environment and **the commit** — neither half is sufficient alone. Written to a TRACKED directory: `logs/` is gitignored, so a snapshot there would be exactly as ephemeral as the thing it describes, and `.gitignore`'s blanket `*.txt` needed an explicit negation, pinned by `test_environment_snapshots_are_not_gitignored` (this repo has lost a file to a blanket rule before — `*.yml` silently swallowed a new workflow). Not fully Resolved: the operator must commit the snapshots, and nothing enforces that. **The first snapshot ever taken immediately diagnosed a live production breakage** — it showed `views-pipeline-core` installed `-e` (editable) with no `views-frames` line, which is why all four production ensembles fail at import; see C-116. |

---

### C-118 — 27 models accept any future views-datafactory major

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | views-datafactory publishing a 2.0. |
| **Source** | expert-code-review (2026-08-02) |
| **Status** | Open |
| **Location** | 27 `requirements.txt` carrying `views-datafactory>=1.9.0` (10 in `envs/views-baseline`, 12 in `envs/views_r2darts2`, 4 in `envs/views-hydranet`, 1 in `envs/views-r2darts2`); the sibling `postprocessors/un_fao/requirements.txt` already carries `views-datafactory>=1.9.0,<2.0.0` |
| **Notes** | An unbounded upper spec on 27 models, installing itself during a monthly hand-run on whichever laptop is free. The 28th file already carries the ceiling, so closing this is making 27 files match a decision the repo has already made rather than making a new one. **Tier 3 and not 2 deliberately:** this is a specific, measured instance of **C-31** (*"upstream algorithm package API changes break views-models silently"*, Tier 2), and the family severity is already carried there — double-counting it would inflate the register rather than inform it. Escalate only if the same pattern is found on a package with no Tier-2 parent. Cross-refs: **C-31** (parent), C-50 (spec unresolvable on fresh clone), C-116. |

---

### C-119 — The install gate decides a production install from a line count of pip's log

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | pip emitting any unexpected line — a deprecation warning, an index warning, a proxy notice — or a pip older than 22.2, which has no `--dry-run` and errors instead. |
| **Source** | expert-code-review (2026-08-02, Feathers/Nygard) |
| **Status** | Open |
| **Location** | `run.sh:27-33` in ~131 scripts, e.g. `models/bad_blood/run.sh:27` |
| **Notes** | The gate is `missing_packages=$(pip install --dry-run -r requirements.txt 2>&1 \| grep -v "Requirement already satisfied" \| wc -l)`, then install when `>0`. It counts **lines of merged stdout and stderr**, not missing packages, so its answer depends on pip's log formatting and on stderr being quiet. Any warning triggers a full `pip install` — which, in a shared environment (**C-116**), can *mutate other models' dependencies as a side effect of a log message*. There is no seam to test it: the logic is inline and duplicated ~131 times, which is **C-70**'s cost made concrete. The fix belongs in the generator, not here — `template_run_sh.py` in views-pipeline-core, already open as **views-pipeline-core#384** for the shebang — otherwise it follows the C-39 pattern of fixing copies while the generator keeps producing the defect. Cross-refs: **C-70** (duplication), **C-116** (shared mutable environment), **C-39** (fix-the-copies-not-the-generator, regressed 24x), views-pipeline-core#384. **Cluster G** (added 2026-10-26, #610) — the gate's input is a line count of pip's log, a proxy it cannot interpret, so any unexpected line flips a production verdict. |

---

### C-120 — A general test-exemption mechanism with a model-specific guard: any model could opt out of parity enforcement silently

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Adding `EXPERIMENT_IN_PROGRESS = True` to any model's `config_hyperparameters.py` — a one-line, plausible-looking edit that exempts that model from the cross-ensemble parity pins. Also: any future exemption mechanism whose *escape hatch* is general while its *alarm* names one subject. |
| **Source** | code-review of `chore/s1_5-fleet-config-hygiene` (2026-08-03), reproduced |
| **Status** | Resolved |
| **Location** | `tests/test_datafactory_parity.py` — `_experiment_in_progress()`, `test_both_trios_use_same_loss`, `test_constituent_posterior_samples_match`, and the guard now named `test_the_experiment_in_progress_roster_is_exactly_as_declared` |
| **Notes** | The branch introduced a correct idea — an experiment whose values churn should not be pinned to an exact value that flickers red/green — and guarded it with `test_violet_visitor_is_experiment_in_progress`, which asserted only that **violet_visitor** carries the marker. **But `_experiment_in_progress()` applies to any model.** So the exemption was general and the alarm was specific. **Reproduced, not theorised:** adding one line to `models/pink_pirate/configs/config_hyperparameters.py` removed it from *both* parity pins and the suite reported **51 passed** with nothing to indicate a model had stopped being checked. A second probe marked all six trio models: both pins then compared `{} == {}` and passed — a test asserting nothing at all. **Fix:** pin the exemption **roster as a set** (`EXPERIMENTS_IN_PROGRESS = {"violet_visitor"}`), which fails in both directions — a model gaining the marker and a model losing it — plus an explicit non-empty assertion in each pin so neither can ever pass vacuously. Verified by re-running all three probes against the fix. **The generalisable rule:** *an escape hatch must be guarded at the same scope it operates.* A guard that names one subject cannot cover a mechanism that accepts any. Cross-refs: **C-112** (a check asked in a scope that cannot answer it), **C-113** (a gate that cannot tell "clean" from "did not run"), **C-115** (an invariant guarded by name rather than by the invariant). Member of **Cluster A** (declared-but-unenforced). ⛔ **RECURRED 2026-10-18 — see C-163.** This rule was broken three ways in one change to `tests/test_roster_conformance.py`, the sibling of the file that cites it. A rule stated as generalisable in a register entry is not enforced by anything; C-163 records what it takes to enforce it. **Near-miss recorded on C-71:** the same branch put prose after `Open` in a Status field, which silently dropped the entry from the header count — caught by `test_open_count_accurate`, which is what that test is for. |

---

### C-121 — The delivery boundary accepts a forecast of unbounded age

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | `un_fao` running in a month where no fresh `rusty_bucket` run was produced first — which is every month, because `rusty_bucket` is not in `monthly_run.sh`. |
| **Source** | expert-code-review of the delivery composition (2026-08-04), traced against the live store |
| **Status** | Open |
| **Location** | `views-postprocessing/views_postprocessing/unfao/managers/unfao.py::_read_forecast_data_contract`; `contract/wire/source_selection.py::resolve_run` |
| **Notes** | `resolve_run` selects **the newest fully-manifested run for a named ensemble** — identity plus completeness, which is right, and is what **C-97** records as resolved. What it does not do is bound the run's **age**. So the delivery step will silently republish an arbitrarily old forecast as the current one. **Measured 2026-08-04:** FAO's forecast stream is 145 days stale (#320, newest `forecast_dataset` 2026-03-10) while `production_forecasts` holds exactly one complete run, `rusty_bucket_forecasting_20260727_095355` (all three `lr_ged_*` targets), untouched since 27 July. The forecast existed; nothing carried it across, and nothing would have objected if it had carried the March one instead. The method **already logs** `"Contract inbound resolved: run %s"` — the fact needed for the assertion is in hand and simply not asserted on. **Exit:** a declared freshness budget and a refusal, at the boundary that already knows the answer. This is the one change that addresses the failure that actually occurred. Cross-refs: **C-99** (no missed-month signal — this is its specific, now-measured instance at the delivery boundary), **C-97** (selection, resolved), **C-114** (a detector that reported its own failure as mild staleness). Member of **Cluster A** (declared-but-unenforced). |

---

### C-122 — The production pipeline's assembly is five ordered strings, and the order is an unstated data dependency

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Adding a producer *below* its consumer in `monthly_run.sh`, or reordering the existing five lines. Concretely: adding `rusty_bucket` after the `un_fao` line rather than before it. **Extended 2026-08-25 (#425):** also **arming a delivery without adding its launcher to the list** — the same hand-kept-list root cause, failing by omission rather than by order. |
| **Source** | expert-code-review of the delivery composition (2026-08-04) |

> **Second symptom, observed 2026-08-25 (#425): omission, not just order.** `deliveries/un_crafd.py` has
> been `live(since=2026-08-14)` since #399, and `postprocessors/un_crafd` **is not in `monthly_run.sh` at
> all** — so there is an armed, production-tier delivery the monthly path cannot run. `rusty_bucket`, which
> `un_fao` is declared to consume, is likewise absent. The register's earlier summary of this composition —
> *"produces four forecasts nobody delivers and delivers one forecast nobody just made"* — now has a third
> clause: and does not deliver one that is armed.
>
> Same root cause, which is why this is an extension rather than a new entry: the list is typed, not derived.
> ADR-019 §3 and §7 claimed `frequency` had already made it *"a filter over declarations"*; measured, it
> contains **zero** references to `deliveries/` or `frequency`. That claim was corrected in the ADR by #425;
> **building the filter is not done, and is what would close this entry rather than patch it.**
| **Status** | Open |
| **Location** | `monthly_run.sh` — the five `run_folder` lines; `postprocessors/un_fao/configs/config_meta.py` (`"ensemble"`); `views-postprocessing/unfao/product.py` (`UPLOAD_ENABLED`) |
| **Notes** | `un_fao` consumes what the ensembles produce, and that dependency is encoded **only** as line order in a shell script. Getting it wrong raises nothing: the consumer simply delivers a previous run. **Everything beneath this point injects its dependencies** — `_ContractStorePort` is an explicit DIP port, `contract/` and `delivery/` are partner-neutral and a test proves it — while the composition root hard-codes five concrete paths in a fixed sequence. The three files that must agree to deliver a forecast live in two repositories with no reference between them. Related: **13 ensembles exist and only 4 are in `monthly_run.sh`**; the one `un_fao` is configured to consume, `rusty_bucket`, is **not among them**, so "run everything monthly" produces four forecasts nobody delivers and delivers one forecast nobody just made. **Deliberately not fixed yet.** A declared composition is an abstraction over exactly one instance in this repo, and the maintainer's rule is to extract on a second incident behind a named trigger. **The named trigger: when a second partner delivery needs a production run in views-models.** The scar already exists one repo away — views-postprocessing #211, *"every partner-scoped guard was scoped to ONE partner"*, fixed there with a declared list asserted against the filesystem (`tests/conftest.py::PARTNER_PACKAGES`), not a framework. That is the shape to copy when the trigger fires. Cross-refs: **C-99**, **C-97**, **C-121**, **C-120** (same bug class: a general mechanism guarded for one subject). |

---

### C-123 — `rusty_bucket`'s config does not describe what it emits

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Any tool or person deriving a delivery's target names from model config rather than from the manifests in the bucket. |
| **Source** | expert-code-review (2026-08-04); confirmed against the live store |
| **Status** | Open |
| **Location** | `ensembles/rusty_bucket/configs/config_meta.py:4` |
| **Notes** | Declares `regression_targets: ["lr_sb_best", "lr_ns_best", "lr_os_best"]`. The run it actually produced emitted `lr_ged_sb`, `lr_ged_ns`, `lr_ged_os` — verified by listing `production_forecasts`, where the manifests are named `rusty_bucket_forecasting_20260727_095355__lr_ged_*__manifest.json`. Delivery works **only** because `resolve_run` matches manifest filenames rather than the config. So the config is decorative at precisely the point a reader would trust it, and the only reliable source of truth for a delivery's contents is a live bucket query — which is what this review had to do. Cross-refs: views-models#151 (target-name standardisation), **C-104** (one quantity, four config keys). |

---

### C-124 — Coverage is validated after the expensive path, not at resolution

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A run resolving successfully for a region whose cell coverage it cannot satisfy — e.g. an africa-only run against `region: land_gaul`. |
| **Source** | expert-code-review (2026-08-04) |
| **Status** | Open |
| **Location** | `views-postprocessing/contract/wire/source_selection.py` — `resolve_run` (manifests) vs `TargetLease.load()` (`assert_complete_coverage`, `assert_no_excluded_cells`) |
| **Notes** | `resolve_run` checks that every expected target has a content-verified manifest; `expected_cells` and `excluded_gids` are passed to the lease and only enforced when frames materialise. Resolution succeeding therefore does not mean delivery will succeed — the failure arrives after shard downloads, on a monthly hand-run on a laptop. The lazy design is correct for memory (it is the run-0 OOM fix); what is missing is a cheap precheck so an unsatisfiable run is rejected before the heavy fetch. Cross-refs: **C-121**. |

---

### C-125 — A target-name gate would fail correct delivery files, because a source's config does not describe what it emits

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Enabling an edit-time `targets` check in a delivery declaration (ADR-019 §1 `REQUIRE.targets`) while any source's config still misdescribes its own output. |
| **Source** | expert-code-review of the delivery declaration design (2026-08-04) |
| **Status** | Open |
| **Location** | `ensembles/rusty_bucket/configs/config_meta.py:4`; the proposed `deliveries/*.py` `REQUIRE.targets` (ADR-019) |
| **Notes** | `rusty_bucket` declares `regression_targets: ["lr_sb_best", "lr_ns_best", "lr_os_best"]` and emits `lr_ged_sb/ns/os` (**C-123**). A `targets` gate checked against the source config would therefore reject a *correct* delivery file for the repo's own FAO ensemble. **The cost is pedagogical, which is why it is worth an entry of its own:** ADR-020 makes error messages load-bearing — the design's value is that a newcomer is guided down one level at a time — and the first lesson this would teach is that the repo's errors are wrong. Nothing recovers from that. **Exit:** fix C-123, then promote `targets` from a run-time assertion (checked against the run's manifests, which are truthful) to an edit-time one. Until then ADR-020 §4 records `targets` as a stair that ends outside the repo. Cross-refs: **C-123**, ADR-019 §1, ADR-020 §4. |

---

### C-126 — The delivery design makes dormancy visible but not absence of execution

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A delivery declared `intent = live()` that no runner picks up, or that ships a run far older than the consumer expects. |
| **Source** | expert-code-review of the delivery declaration design (2026-08-04, Nygard) |
| **Status** | Open |
| **Location** | proposed `deliveries/*.py` — `DELIVERY.intent`, `REQUIRE.max_age`; ADR-019 §4 freshness rule, ADR-020 §4 "where the stairs end" |
| **Notes** | The design solves the *paused* case well: `paused(reason, since=...)` cannot be set silently, so a dormant edge carries an explanation and an age. It does **not** solve the *live-but-never-run* case — a `live()` delivery that nothing executes produces no error, because nothing failed. That is precisely the 145-day FAO silence (**C-121**, **C-99**), surviving the redesign. ADR-020 §4 names it honestly as *"not a locked door — a hole in the floor"*. **Exit, two parts, both already written into the amendment:** `REQUIRE.max_age` is mandatory whenever `intent = live()` (ADR-019 §4), and `tools/liveness` reports **derived status beside declared intent** (ADR-017 §7), so `live()` + "never delivered" is a visible contradiction rather than an absence. Registered separately from C-121 because C-121 is the defect in today's code and this is the residual risk in tomorrow's design — closing one does not close the other. Cross-refs: **C-121**, **C-99**, **C-110**. |

---

### C-127 — Fleet-wide config values are written in two literal styles, so a naive grep silently under-counts

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Executing the `deployment_status` → `maturity` migration (ADR-017 §11), or any future statement of the form "measured across all N sources" produced by grepping the fleet. |
| **Source** | falsify audit of the four delivery documents (2026-08-04), probe P1 |
| **Status** | Open |
| **Location** | 128 × `{models,ensembles}/*/configs/config_deployment.py` — 47 use `{"deployment_status": "shadow"}`, 81 use `{'deployment_status': 'shadow'}` (e.g. `models/brown_cheese/configs/config_deployment.py:19`); the false figures were at `docs/ADRs/017_source_composition_delivery.md:74` and `docs/forecast_delivery_map.md:158` |
| **Notes** | ADR-017 §2 and the delivery map both stated *"Measured across all 131 sources: 120 `shadow`, 6 `baseline`, 4 `deprecated`, 1 `deployed`"*. The true distribution is **117 / 6 / 4 / 1 across 128 files**, and there are **132** source directories — four (`models/cool_cat`, `models/teenage_dirtbag`, `models/test_model`, `ensembles/test_ensemble`) carry no `config_deployment.py` at all. The measurement had been taken with a double-quote pattern that matched 47 of 128 files and silently reported the rest as absent. **The wrong number is now corrected in both documents; the durable risk is the heterogeneity that caused it.** ADR-017 §2's numbers are its evidence base, so this was not cosmetic — a rule whose stated justification is false is the kind a future contributor overturns. Same bug class as **C-114** (a detector blind to one spelling of the thing it was built to see), in a different subsystem: there a rejected key read as mild staleness, here 81 configs read as non-existent. **Exit:** the migration tool must parse rather than grep, and must assert it touched 128 files. Cross-refs: **C-114**, **C-124**. ⚠️ **Second mechanism, found in #648 (2026-10-30): a count can under-count by spelling, and it can also be blind to WHICH instance it counted.** `tests/test_crafd_chain_runner.py` asserted `script.count("DERIVED") >= 2` on the premise that both of `pod_run_crafd_chain.sh`'s floors were reasoned rather than observed. The memory floor was measured on 2026-10-28 and the comment block above it was updated to say so, while the line the operator actually reads still printed `floor 16GB, DERIVED not measured` — and the guard held that label in place, because two occurrences satisfied `>= 2` whichever floor they belonged to. **The guard survived its own subject's correction**, which is the sibling of C-112's instance 2 (a test that enforced the bug it was meant to prevent). Fixed by making the check per-floor and two-directional: the measured floor must say measured, the derived one must say derived, and mislabelling in either direction is red. Cross-ref **C-112**. |
---

### C-128 — ADR-019's "REQUIRE only refuses" rule is false for its one mandatory key, so a validator author must guess

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Implementing the `REQUIRE` validator, or writing the first real delivery file under ADR-019. |
| **Source** | falsify audit of the four delivery documents (2026-08-04), probe P3 |
| **Status** | Open |
| **Location** | `docs/ADRs/019_delivery_declaration.md` §2 (the testable rule, and the "may be omitted entirely" allowance) vs §4 (Freshness) |
| **Notes** | §2 states the rule that defines the whole two-block design: *"removing a line from `REQUIRE` must never change what is produced, only what is allowed through."* §4 then requires that a `live()` delivery **must** declare `max_age`. Removing `max_age` does not widen what is accepted — it makes the file invalid, so nothing is produced. A second instance sits in the same paragraph: §2 offers *"`REQUIRE` may be omitted entirely when there is nothing to assert"*, but every `live()` delivery must carry `max_age` and every real delivery is live, so the allowance is never actually available. **Why this is Tier 2 rather than a wording nit:** an implementer who resolves the contradiction toward "`REQUIRE` is purely assertive" will not enforce freshness — and the missing freshness bound is precisely the failure that already occurred (**#320**, 145 days of silent non-delivery). The contradiction points the implementation at the bug. **Now corrected in the ADR**; the entry records the class so the next absolute-sounding rule is checked against its own exceptions. Cross-refs: **C-121**, **C-126**, **D-09**. |

---

### C-129 — The delivery declaration has no home for the key that actually arms the delivery

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Building `deliveries/un_fao.py` (ADR-017 §11 Phase 1), or adding the second consumer under **#333**. |
| **Source** | falsify audit of the four delivery documents (2026-08-04), probe P8 (adequacy) |
| **Status** | Open |
| **Location** | `postprocessors/un_fao/configs/config_meta.py:26-27` (`wire_contract`, `wire_upload_enabled`); `docs/ADRs/019_delivery_declaration.md` §3 (the key set) |
| **Notes** | The real FAO config declares **eight** keys. Five map cleanly onto ADR-019's schema; three had no home — `algorithm`, `wire_contract`, and `wire_upload_enabled`. The last is the **arming switch**: views-postprocessing ADR-013 §11.4 sets `UPLOAD_ENABLED = False` and makes that launcher key its only override. ADR-019 mentioned it **zero times** while the delivery map mentioned it twice. The consequence is exact and self-inflicted: ADR-019 exists because a delivery-deciding line sits buried in a file whose docstring calls itself inert — and the design moved the `ensemble` line out while **leaving the on/off switch behind in that same file**. Worse, `intent = live()｜paused()` and `wire_upload_enabled: True｜False` are the same fact in two places, which is the duplication ADR-019 §8 rejects by name. **Resolved in the ADR by declaring `intent` the repo-side arming switch, from which the launcher key is *derived*** — derivation, not duplication, the same principle as the filename carrying the consumer. `wire_contract` is recorded as an unconditional constant (the legacy leg retired in **#149**) and `algorithm` as framework plumbing that stays put, so all eight keys now have a stated home. **The residual risk this entry tracks:** until `deliveries/` is built, the two switches co-exist and can disagree. Cross-refs: **C-110** (the switch exists only in an uncommitted tree), **C-63**, **C-126**. |

---

### C-130 — The maturity migration leaves ten sources with no destination value

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Executing the `deployment_status` → `maturity` rename across the fleet (ADR-017 §11 Phase 2). |
| **Source** | falsify audit of the four delivery documents (2026-08-04), probe P4 |
| **Status** | Resolved (2026-09-17, #479) |
| **Location** | the 6 sources declaring `baseline`; `models/cool_cat`, `models/teenage_dirtbag`, `models/test_model`, `ensembles/test_ensemble` (no `config_deployment.py`) |
| **Notes** | ADR-017 correctly holds that `baseline` is a **role**, not a maturity, and that it leaves `config_deployment.py` entirely — but every source still needs *some* maturity, and the migration table originally sent `baseline` to *(nothing)*. Six real sources carry it, and a research assistant migrating the fleet would have had no value to write. A further four source directories carry no `config_deployment.py` at all, so they have no `deployment_status` to migrate *from*. **Both now resolved in ADR-017's migration table** (`baseline` → `candidate`, role preserved where it already lives; the four unconfigured sources named explicitly). Entry retained because the migration is not yet executed and the table is the only thing that makes it mechanical. **2026-09-17:** every reader now applies the table (#476, via `deliveries.coherence.maturity_of`); the rename itself is the follow-up PR, per source and gated on the engine's pipeline-core floor (ADR-017 §11 status 2026-09-17). Cross-refs: **C-127**, **C-147**. |

---

### C-131 — "Production-tier consumer" reads as a discriminating condition while `tier` has one value

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | A reader deriving `is_in_production` from ADR-017 §4e without also reading ADR-019 §3; or the addition of a second `tier` value. |
| **Source** | falsify audit of the four delivery documents (2026-08-04), probe P2 |
| **Status** | Open |
| **Location** | `docs/ADRs/017_source_composition_delivery.md` §4e (the ⟺ definition) and §5 (the tier rule) vs `docs/ADRs/019_delivery_declaration.md` §3 (`tier`) |
| **Notes** | §4e defines *in production ⟺ maturity is `graduate` **and** a delivery ships it to a **production-tier** consumer.* ADR-019 §3 fixes `tier` at exactly one value, `prod`, and says plainly that the gate is therefore *currently unconditional*. §4e did not, so the definition read as two independent conditions when one is presently always true. Not a correctness defect — the definition stays true as written, and becomes discriminating the moment a second tier value exists — but it is the kind of gap that makes a newcomer believe a check exists that does not. **Corrected by a clause in §4e pointing at ADR-019 §3.** The second tier value is itself blocked on ADR-017 §12's open shadow-destination question. Cross-refs: **C-129**. |

---

### C-133 — The coverage copy the manager actually reads was the one nothing checked

| Field | Value |
|---|---|
| **Tier** | 1 |
| **Trigger** | Editing `coverage` in `deliveries/un_fao.py` or `REGION` in `config_queryset.py` without also editing `config_meta.py`'s `"region"` — or any consumer cloning the un_fao configs, which is what #333 was about to do. |
| **Source** | expert-code-review of the proposed ADR-021 (2026-08-11), found while verifying #373 |
| **Status** | Resolved (2026-08-11, ADR-021) |
| **Location** | `postprocessors/un_fao/configs/config_meta.py` (was `"region": "land_gaul"`); consumed at `views-postprocessing unfao/managers/unfao.py:236,312,419`; written at `delivery/provenance.py:47` |
| **Notes** | `land_gaul` was a typed literal in **three** places: the delivery declaration, `config_queryset.REGION`, and `config_meta["region"]`. `_upload_armed()` cross-checked the first two and disarmed loudly on disagreement — verified working, a clean checkout yielded `wire_upload_enabled = False`. It never checked the third, **and the third is the only one that leaves this repository**: the un_fao manager reads `configs.get("region")` and writes it into the provenance record delivered to the UN FAO. Setting the two guarded copies to `"land"` **arms** the upload while the consumed copy still says `"land_gaul"` — the run curates to `land_gaul` and ships provenance claiming `land_gaul` against a declaration saying `land`. No warning, plausible output, external partner. Tier 1: silent, wrong, and UN-facing. **The general shape: a reconciliation is not a derivation.** Cross-checking two copies leaves the duplication in place and implies all copies are covered; here it covered the two that did not matter. **Resolved by derivation, not by extending the check to a third copy** — hardening a duplication is the direction ADR-019 §8 already rejected. All three now read `deliveries.status.declared_coverage()`; the cross-check and its `ast` parser are deleted; a one-line assertion remains as belt-and-braces and is verified to bite. Cross-refs: **C-110** (the same region uncommitted for seven weeks, closed by #127), **C-129** (same fact in two places), **C-57** (`_queryset_region()` parsed a sibling's source text for a literal — same family). Member of **Cluster A** (declared-but-unenforced). **Cluster G** (moved from Cluster A, 2026-10-26, #610) — the interlock ran and passed; its input was the two copies the manager never reads. Resolved by ADR-021, kept as precedent. |

---

### C-134 — The partner launcher is now cloned twice, and the shared half is production infrastructure

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | **A third partner launcher**, or **the first bug hand-patched in both** existing ones. Either is the signal that the current split (shared protocol, per-partner configs) has stopped describing reality. |
| **Source** | views-crafdapi#43's design constraint, executed while building `un_crafd` (#333) |
| **Status** | Mitigated (ADR-022) |
| **Location** | `tools/launcher/postprocessor.sh` (the shared body); `postprocessors/un_fao/run.sh` and `postprocessors/un_crafd/run.sh` (the wrappers); per-partner copies remain in each `configs/` and `main.py` |
| **Notes** | Adding `un_crafd` made the partner-launcher pattern n=2 in this repository. The register already noted at the `crafd/` clone entry that the trigger had fired for **views-postprocessing** and not for views-models; #333 is the event that fires it here. **What was extracted, and why that half and not the other:** `run.sh` changes for two unrelated reasons — because a partner differs (conda env, pin) and because the delivery *protocol* differs (registry before conda, environment after it, capability by import not grep). The second must not be copied: a protocol fix would need hand-applying per partner, and the first one missed fails silently. That is not hypothetical — views-postprocessing cloned `unfao/` into `crafd/` and shipped a follow-up whose message is *"every partner-scoped guard was scoped to ONE partner"* (their #211, their C-33). `un_fao/run.sh` went 136 lines → 21. **What was deliberately NOT extracted:** `configs/*.py` and `main.py`. Their values genuinely differ per partner, and the ones that must agree platform-wide are already derived from `deliveries/<consumer>.py` (ADR-019, ADR-021). Two partners is the trigger to share what must not vary, not a licence to abstract what must. **The residual risk this entry exists for:** the shared body is sourced by production launchers, so editing it changes the live FAO delivery whether or not FAO appears in the diff. It deserves `platform_env.sh`-grade care. Mitigated by the launcher guarantees being parametrised over **every** launcher and reading each one's *effective* text (wrapper + sourced body) — reading `run.sh` alone would have made all of them pass vacuously the moment the body moved. Cross-refs: **C-60** (grouping by responsibility), C-57 and C-112 (scars the body carries), ADR-018 (the same one-writer shape a layer down). |

---

### C-135 — The live FAO delivery runs on a build whose upload check fails OPEN

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Any armed delivery run while `VIEWS_POSTPROCESSING_PIN` resolves to a views-postprocessing build predating their #222 — which is every merged branch today. Acute on `un_fao`, which is `live`. |
| **Source** | views-postprocessing's review of views-models#380 (2026-08-11), verified here against their tree |
| **Status** | Resolved |
| **Location** | `postprocessors/un_fao/run.sh` (`VIEWS_POSTPROCESSING_PIN="1.1.0"`, was `"main"`); upstream `views_postprocessing/{unfao,crafd}/managers/*.py` |
| **Notes** | Until views-postprocessing#222 the upload verification read `if success is False: raise`. That **fails open**: a result that is `None`, lacks the attribute, or carries a non-bool passes as though the upload succeeded. The consequence is an **orphan** — a file in the partner bucket with no metadata document — and both consumer APIs select on metadata, so the partner sees nothing while the run reports success and exits 0. Their fix is `if success is not True`. **Why it cannot simply be fixed here:** the fix lives on their `development` (`2eb29f1`), not on `main` (`3286eab`). `main` is the newest merged state carrying the crafd package at all — the only tag, `1.0.0`, predates it and contains zero crafd files. So **no merged pin exists today with both crafd and C-79**, which is why **#364** (pin to a released tag) now blocks two repositories rather than one. Moving a live delivery's pin to an unmerged branch is a worse trade than waiting for the sync, so this is registered rather than patched. **`un_crafd` is not exposed**: it is `paused`, so the manager never constructs a partner store and makes zero store calls — the gap needs an upload to reach. **Exit:** views-postprocessing merges `development` → `main` (the same action that lets them cut a tag), then both pins move and `tests/test_launcher_pin_safety.py`'s strict xfail flips to XPASS, forcing the marker's removal. Guarded meanwhile by that file: an armed launcher on a deficient pin fails, so the arming and the pin must move together. Cross-refs: **C-134** (the shared launcher body where the pin now lives), #364, #333. **RESOLVED 2026-08-13 (#391).** views-postprocessing cut tag **`1.1.0`** (`1e21d723`, 06:33) carrying both the crafd package and C-79; both launchers moved to it. Not taken on trust — verified in the installed prefix *before* the armed run: `unfao/managers/unfao.py:71` reads `success is not True`, and `delivery/provenance.py:57` carries `DESCRIPTION_MAX = 255`, the bound whose absence produced the real orphan at 19:41 on 2026-07-27. The delivery then landed **111 files with 111 metadata documents and zero orphans**, confirmed by querying the bucket and the `file_metadata/unfao` collection directly rather than trusting the run's exit code. The strict xfail XPASSed on the pin move and was removed, exactly as designed. **Residual, registered separately as C-139:** arming is per-consumer but the conda prefix is shared, so a *disarmed* launcher on a deficient pin can still downgrade the build an armed one uses. **Cluster G** (added 2026-10-26, #610) — a check that fails OPEN is the canonical form of the cluster: the verdict is clean precisely when the check could not be performed. Resolved, kept as precedent. |

---

### C-136 — The pandas-free evaluation path is built, tested and unreachable: `data_format` defaults to `dataframe` and no model declares it

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | The Epic #242 roster is switched from `land` to global `land_gaul` coverage and evaluated on the server (S5) — the actuals load then pulls ~28.4M rows through pandas on a box that is already holding the model. Also fires the first time anyone tries to *use* the pandas-free branch and discovers it has never run in production. |
| **Source** | cradle-to-grave pandas audit, 2026-08-12 (three parallel sweeps + live process inspection of the `blazing_meteor` training run) |
| **Status** | Open |
| **Location** | views-pipeline-core `modules/dataloaders/datafactory_contract.py:40-41` (absent key → `dataframe`), `managers/evaluation/stage.py:53` (the default), `:112`, `:288-299` (`read_dataframe` on the actuals), `modules/validation/adapter.py:308-349` (the pandas→numpy reduction); the **unreachable alternative** at `stage.py:170-257` (`_load_actuals_frame`) + `adapter.py:351-399` (`from_actual_arrays`); consumers: `models/*/configs/config_queryset.py` — **zero** `data_format` declarations across all 131 models |
| **Notes** | The frame-native actuals path (#301/#302) is fully implemented, pandas-free, and has **no production caller**. It is selected by `data_format` on the queryset, and `declared_data_format` returns `dataframe` when the key is absent — so the branch is off not because anyone chose pandas but because **nobody typed a key**. The only two `data_format` declarations in this entire repository are `postprocessors/un_fao/configs/config_queryset.py:78` and `postprocessors/un_crafd/configs/config_queryset.py:77`. Every HydraNet run, including all eight Epic #242 constituents, takes the pandas branch of live code that has a working alternative sitting beside it. **Not a correctness risk — a scale ceiling and a false-assurance risk.** The output is right; what is wrong is that "the platform is going pandas-free" is true of the delivery leg and false of the evaluation leg, and nothing in the repository says so. `rusty_bucket` pins the pandas branch harder still: `prediction_frame_ensemble.py:531-539` builds its `EvaluationContext` without `data_format`, and `evaluation/stage.py:103-107` refuses `feature_frame` for ensembles outright ("Frame-fed ensemble constituents are not supported yet"). **The delivery is not exposed by this** — the postprocessors call `get_queryset()` on their own path manager, and absence there is a *refusal*, not a fallback (`views_postprocessing/contract/launch_config.py:117` — *"there is no fallback to omit your way into"*). **Priority, per the maintainer (2026-08-13): get pandas to the seams first, then out.** That ordering is what this entry tracks — the seam here already exists and is one key wide, so declaring `data_format` on the model querysets is the cheapest possible first move and is a prerequisite for the second. **Exit:** (a) declare `data_format` on the HydraNet querysets so the frame branch runs at all, which also gives it its first production exercise; (b) lift the ensemble refusal at `stage.py:103-107`. Note that pipeline-core reached the same verdict independently and recorded it in `tests/test_falsification_engines_pandas_free.py` (*"VERDICT: FALSIFIED — 4 hard"*), env-gated behind `RUN_PANDAS_FREE_E2E=1` so it never fails CI — a known-true finding that cannot nag anyone, which is the C-113 shape. Cross-refs: **C-137** (the fetch leg, which has no seam to declare), **C-138** (the import leaks), **C-89** (the same migration, reconciliation leg), C-113, C-99, views-postprocessing C-56 (the producer OOM this pattern caused, now Resolved), views-faoapi C-191 (the serving-side twin, still open). |

---

### C-137 — HydraNet's input path is pandas from zarr to the volume, and there is no frame-native replacement to switch on

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | views-hydranet#157 is scheduled and scoped from the assumption that a FeatureFrame path partly exists — it does not; the repository has **zero** `zarr` or `FeatureFrame` references, so this is a build, not a switch. Also fires whenever the `[pandas-free PredictionFrame path]` log line is read as a statement about the run rather than about the output container. |
| **Source** | cradle-to-grave pandas audit, 2026-08-12 (verified empirically: 212 pandas and 121 pyarrow shared objects mapped into live training pid 3161064) |
| **Status** | Open |
| **Location** | views-hydranet `views_hydranet/utils/grid_to_dataframe.py:87` (`pd.DataFrame`), `data/data_fetcher.py:60` (re-reads the parquet), `data/volume_handler.py:325` (`vol[...] = df[col].values` — the df→numpy boundary), `data/feature_scaler.py`; views-pipeline-core `modules/dataloaders/dataloaders.py:600-742`; artifact `models/<name>/data/raw/calibration_datafactory_df.parquet` (the `_df` form, not `_ff`) |
| **Notes** | The fresh Epic #242 fetch wrote the pandas artifact, and the same parquet is read through pandas **three times** on one run: building it, loading it for training, and again for actuals at evaluation. Everything downstream of `volume_handler.py:325` — `training_engine.py`, `hydranet_inference.py`, the PredictionFrame — is genuinely clean, which is why the log line reads as it does; it describes the *output container*, not the lifecycle. **Sizing, so the risk is not overstated:** `bold_comet`'s parquet is 5,034,240 rows × 6 cols = 262 MB in pandas, projecting to roughly 1.3 GB at `land_gaul`. The model-side fetch is therefore **not** where the memory ceiling lives — the ceiling is at evaluation (C-136) and at serving (views-faoapi C-191, 24 GB co-hosted box). This entry is registered as a **migration and honesty risk**, not a memory one. **Why separate from C-136:** C-136's seam exists and is one key wide; here there is no seam at all and three files must be written. Different exit, different repository, different owner. **Exit:** views-hydranet#157 — now **unblocked**, its prerequisites pipeline-core #161 and #162 both closed. It names the three files above. Under the maintainer's stated ordering (seams first, then out), this is the *second* move: C-136 declares the seam, this removes the pandas behind it. Cross-refs: **C-136**, **C-138**, C-89, Epic #242, views-hydranet#157, views-pipeline-core Epic #265 / #268. |

---

### C-138 — Two eager re-exports pull pandas into every "pandas-free" process, and the purity test probes neither

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Someone cites `tests/test_import_purity.py` as evidence that the PredictionFrame path is pandas-free — for `views_pipeline_core.managers.ensemble` (which is `ensembles/rusty_bucket/main.py:3`'s own import) and for `managers.prediction.savers` it proves nothing, because the test never imports them. |
| **Source** | cradle-to-grave pandas audit, 2026-08-12 |
| **Status** | Open |
| **Location** | views-pipeline-core `managers/ensemble/__init__.py:1` (eager `DataFrameEnsembleManager` re-export → `dataframe_ensemble.py:20 import pandas`), `managers/prediction/prediction_frame_converter.py:15` (module-scope `import pandas`, reached unconditionally via `savers.py:22-25` and `model.py:623`), `tests/test_import_purity.py`; consumer `ensembles/rusty_bucket/main.py:3` |
| **Notes** | Neither import executes pandas code on the PF path — pooling is `np.concatenate` (`prediction_frame_ensemble.py:119`) and `to_arrow_table` is pyarrow-only. The cost is not runtime; it is that **the claim and the check have drifted apart**. `test_import_purity.py` asserts pandas-freeness for `import views_pipeline_core` and for `managers.model.ForecastingModelManager`, and both of those still hold — so the test stays green while two modules that every PF and PFE run actually imports pull pandas in. `managers/__init__.py` and `managers/prediction/__init__.py` were given lazy `__getattr__` treatment under #320; `managers/ensemble/__init__.py` was not. Also note `prediction_frame_converter.py` has no `from __future__ import annotations`, so the `pd.DataFrame` annotation at `:157` is evaluated at runtime too. **Tier 3 rather than 4** because the risk is the false assurance, not the import: a future "the ensemble path is pandas-free" assertion would be made on a test that does not cover the ensemble path. Same shape as C-113 (a gate that cannot tell "clean" from "did not run"). **Exit:** extend the purity probe to `managers.ensemble` and `managers.prediction.savers`, then make it pass — lazy `__getattr__` on the ensemble package, and move the converter's pandas import inside the two functions that use it. Cross-refs: **C-136**, **C-137**, C-113, views-pipeline-core #320. **Cluster G** (added 2026-10-26, #610) — the purity test never imports the two eager re-exports, so its input excludes the only path by which the impurity arrives. |

---

### C-139 — Arming is per-consumer, but the conda prefix is shared: a paused launcher can downgrade the build an armed delivery runs on

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Anyone runs a **disarmed** postprocessor that shares a prefix with an armed one — concretely, the views-crafdapi **D4 dry run** (#44), which is planned and whose whole point is to execute `un_crafd` before arming it. |
| **Source** | review-diff on #391 (2026-08-13) |
| **Status** | Mitigated |
| **Location** | `postprocessors/un_fao/run.sh:10`, `postprocessors/un_crafd/run.sh:10` (both `POSTPROCESSOR_ENV_NAME="views-postprocessing"`); `tools/launcher/postprocessor.sh:92-93` (the unchecked pip line); `tests/test_launcher_pin_safety.py` (the guard) |
| **Notes** | Both postprocessors pip-install into the **same conda prefix**, but `intent`/arming is declared **per consumer**. So the safety property "an armed delivery must not run on a fail-open build" was being enforced at the wrong scope: `test_a_disarmed_launcher_stays_disarmed_while_its_pin_is_deficient` correctly passes for a `paused` consumer on a bad pin — and uselessly, because the danger is not to that consumer, it is to its **neighbour**. Concretely, before #391 `un_crafd` pinned `3286eab` (a build in this file's own `DEFICIENT_PINS`, lacking C-79) while `un_fao` was `live`. Running `un_crafd` — the D4 dry run — would have **downgraded the prefix the armed FAO delivery then uses**. Two things make that silent rather than loud: `tools/launcher/postprocessor.sh` has no `set -e` and no `\|\| return 1` on the pip line, so a later reinstall can fail without stopping the run; and the #294 capability assertion still passes on the stale build, because that build also carries `contract/wire`. The end state is C-135 reintroduced on a UN-facing delivery, reported as success. **The PR's own reasoning already contained the argument** — it pins numpy in *both* requirements files with the note "one shared prefix (C-116), so pinning only one lets the other upgrade numpy out from under it" — and then did not apply it to the views-postprocessing pin itself. **Mitigated, not Resolved:** #391 moves `un_crafd` to `1.1.0` and adds `test_no_launcher_can_downgrade_a_shared_prefix_under_an_armed_one`, which asserts that *if any consumer on a prefix is armed, every launcher writing to that prefix must be on a sound pin* — verified to fail when `un_crafd` is re-pinned to the deficient commit. What is **not** fixed is the structure: the prefix is still shared, the pip line is still unchecked, and a third partner inherits the same shape. **Exit:** either a prefix per consumer (costly, and C-116 argues against proliferating environments), or `|| return 1` on the installs in the shared launcher body so a failed reinstall cannot be silent. Cross-refs: **C-135** (the fail-open this would reintroduce, now Resolved), **C-116** (the general shared-environment/run-order concern this is a sharp instance of), **C-134** (the shared launcher body), **C-70**, #364, #333, views-crafdapi #44/#45. |

---

### C-140 — The bound that stops the suite hanging is unverifiable by CI, and its threshold was chosen against a broken backend

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | **When viewser's backend is healthy again**, time one real `ViewserCountryMappingProvider(480, 481).build()` and confirm 90s is comfortably above it. Also fires if anyone reports these tests skipping with "did not return within 90s" while the backend is demonstrably up. |
| **Source** | review of #409 / PR for S1 of epic #412 (2026-08-23) |
| **Status** | Open |
| **Location** | `tests/live_deadline.py` (`VIEWSER_DEADLINE_SECONDS = 90`); the four bounded tests in `tests/test_reconciliation_e2e.py`, `tests/test_reconciliation_viewser_provider.py`, `tests/test_un_fao_datafactory_equivalence.py`; `.github/workflows/run_tests.yml:30` |
| **Notes** | Two residuals from the #409 fix, registered together because they share one cause — **nothing automated ever runs the bounded path.** (a) **CI cannot exercise it.** `run_tests.yml:30` installs only `views_pipeline_core`, `pytest` and `packaging`, so `viewser` and `datafactory_query` are absent and every bounded call short-circuits at `ImportError` before reaching a socket. A green CI is therefore not evidence the bound works; the only machines that execute it are developer machines with the model packages installed, which is the same inversion #409 itself describes. (b) **The threshold is provisional.** 90s was chosen on 2026-08-23 while the backend returned **502 Bad Gateway**, so a *healthy* fetch could not be timed. If a real fetch exceeds 90s these tests skip while the backend is fine — a false negative on a genuine integration guard, and one that reads identically to a real outage. The failure mode is quiet in both halves: the tests skip, the suite is green, and nobody learns the guard stopped guarding. What the fix does establish is bounded and worth keeping separate from the above: the suite completes (404s where it previously ran past a 30-minute kill), and `tests/test_live_deadline.py` proves the mechanism itself — deadline fires, handler restored, timer cancelled, refuses off the main thread, and re-arms so an alarm swallowed by viewser's bare `except:` is not lost — each shown to fail by mutation. Cross-refs: **#409**, **#412** (S1), C-102 (the other place a liveness guard was green and blind). |

---

### C-141 — Two partner delivery surfaces are 95% the same file, so a shared bug costs two fixes

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | **A THIRD partner delivery surface**, or **any bug that has to be fixed in both `unfao_delivery.py` and `crafd_delivery.py`.** Either one is the moment to extract; until then this is a recorded choice rather than an oversight. |
| **Source** | S5 of epic #412 (2026-08-24), measured while adding the CRAF'd surface |
| **Status** | Accepted |
| **Location** | `tools/liveness/unfao_delivery.py`, `tools/liveness/crafd_delivery.py` |
| **Notes** | Measured, not estimated: **14 differing lines in 269** once the consumer name is normalised. The two differ in bucket id, consumer string, class name, surface name and docstring — nothing structural. Copied rather than extracted for three reasons, none of them "it seemed easier". (1) `tools/liveness/README.md:131-140` records the house rule — shared code here was extracted only after **six** surfaces demonstrably duplicated it, and this is the **second** partner surface; WET-before-DRY says duplication is fine at the second occurrence. (2) Extracting would have refactored `unfao_delivery.py` **the same day #416 changed it**, plus its 26 tests, at the end of a five-story sprint — destabilising freshly-verified work for an abstraction the repo's own rule says to wait on. (3) The shape is now *proven* on two live buckets rather than guessed, which means extraction later will be cheap and correct; extraction now would have been both. **The cost is real and already demonstrated:** C-102 (#411) was a bug in exactly this code. Had the CRAF'd surface existed before #416, the fix would have been needed twice — and a copy made *before* #416 would have inherited a matcher reporting `NEVER_DELIVERED` over a healthy delivery. That is why S5 was sequenced after S4. Cross-refs: **#413**, **#412** (S5), **C-102**, `tools/liveness/README.md` (the six-duplications rule). |

---

### C-142 — The CIC sync gate fails on the release PR while the contract it guards is satisfied

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | **Any `development` → `main` release PR** — it is red today on #410 and will be red on the next one. Also fires if anyone reaches for `[cic-skip]` to get past it, which would spend a documented escape hatch (reserved for "purely cosmetic" changes) on a 448-commit release that is not cosmetic. |
| **Source** | release review of #410 (2026-08-24) |
| **Status** | Open |
| **Location** | `.github/workflows/cic_sync_check.yml` |
| **Notes** | The gate reports all nine CIC-governed pairs as missing on #410. **The contract is satisfied.** Verified by running the workflow's own script verbatim, with its own `BASE` (`120ff879`) and the merge commit its log records checking out (`86472880`): 1354 changed files, every governed source paired with its updated CIC, result **PASS**. CI reports FAIL on the same inputs, deterministically across two re-runs — so it is not a race. The failure shape is specific and is the clue: CI evidently sees the *source* files in `CHANGED_FILES` (otherwise nothing would be reported) but not the `docs/CICs/*.md` files, which are demonstrably in the same diff. Cause not established, and this entry does not guess one. **One fact narrows it, observed 2026-08-24:** the same workflow **passes** on every `development`-based PR of this sprint — including #418, which carries this very entry — and fails only on the `main`-based release PR. So the difference tracks the BASE, not the content: whatever `git diff "$BASE"...HEAD` resolves to when the base is a two-month-old `main` is not what it resolves to locally, while the same expression against a recent `development` base is fine. **Why Tier 2 rather than 3:** a gate that fails when the thing it guards is correct trains people to override it, and the override here is a title token that silently disables the check for the whole PR. That converts a real ADR-006 guard into a formality on exactly the changesets — releases — where the most CIC-governed code moves at once. It is also cluster-G shaped: a check believed to be protecting something while its verdict is uninformative. **Second, separable observation:** even working correctly, this gate is a *per-change* rule applied to a *release aggregate*. Every commit in #410 already passed it on its own PR, so re-running it across 448 commits can only produce noise or false failure — arguably the workflow should not fire on `main`-based PRs at all. That is a design question for the workflow's owner, not a fix to make while unblocking a release. Cross-refs: **#410**, ADR-006, `docs/CICs/`. Member of **Cluster G** (a check out of scope returns a clean verdict). |

---

### C-143 — ADR-019 states five coherence rules and `coherence.py` implements them; nothing checks that the two agree

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | **Adding or changing a rule in `deliveries/coherence.py`, or a rule statement in ADR-019 §4.** Concretely: the `provides` coverage rule (#428) and the reconciliation/coverage split (#429) each add or change one, and each is a chance for the pair to drift again. |
| **Source** | S2 of epic #424 (2026-08-26), generalising the #420 falsification |
| **Status** | Open |
| **Location** | `docs/ADRs/019_delivery_declaration.md` §4; `deliveries/coherence.py` (`_check_resolution_and_level`, `_check_target_coverage`, `_check_reconciliation`, `_check_freshness`, `_check_maturity_rules`, `_check_tier`) |
| **Notes** | ADR-019 §4 is a specification of five fail-loud rules. `coherence.py` is their implementation. **Nothing reads the ADR, and no test asserts the two describe the same behaviour** — verified: no test file cites ADR-019 §4, and `docs/validate_docs.sh` checks cross-ADR *references*, not content. The cost is measured, not hypothetical: the #420 audit found six defects in ADR-019 and **three were exactly this drift** — §4's reconciliation rule forbidding a composition the platform needs (HARD 2), §3 documenting two values for a key the code treats as three (HARD 1), and §3/§7 describing a `monthly_run.sh` filter that does not exist (SOFT 4). Each was found by a human reading both, not by a check. **The `None` case is the sharpest illustration:** `coherence.py:187` reads `if require.reconciled is not True`, so unset has always behaved as `False`; the ADR said nothing; and the test suite pinned `False` but **not** `None` — the state that is the default and that both real deliveries are in. S2 (#426) closed that specific hole by documenting the rule and adding the two missing tests, and mutation-proved them: flipping `is not True` to `is False` — precisely the semantics #419 proposed — fails **only** the new test, so the suite as it stood would have accepted that change silently. **What is registered here is the general form, which S2 did not close.** A mechanical check is possible but not obviously worth its overhead — the rules are prose with real nuance, and a shallow "does §4 mention every `_check_*` name" test would be the vacuous guard this register keeps recording. The cheaper discipline, adopted by #424's stories, is that **each story carrying a rule change carries its own ADR edit in the same PR.** Registered so that discipline is visible rather than remembered. **First exercise, 2026-08-26 (#428, S4):** the coverage rule landed with its ADR-019 §4 text in the same PR, and the edit went further than adding a bullet — the module docstring and `TestChecksThatDoNotRunHere` both said `targets` was ungated, which the new rule made *partly* untrue, so both were amended in the same commit. That is the failure mode this entry names, caught by the discipline rather than by an audit six months later. Cross-refs: **#420**, **#424**, **#426**, **#428**, C-122 (the same typed-versus-derived shape in `monthly_run.sh`). |

---

### C-144 — Every delivery coherence rule is an edit-time guard: `check()` has no caller outside the test suite

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | **The first delivery that actually names two sources** — #429 (S5) splits reconciliation from coverage, #430 (S6) lets a multi-source delivery be consumed, and #422 decides which cm coverage to build. When one lands, a composition whose correctness the new rules exist to guarantee becomes reachable, and the guarantee will still only be evaluated by CI. |
| **Source** | S4 of epic #424 (2026-08-26) |
| **Status** | Open |
| **Location** | `deliveries/coherence.py:check()`; callers: `tests/test_delivery_coherence.py`, `tests/test_delivery_errors_descend.py` — and nothing else |
| **Notes** | `check()` runs the six fail-loud rules over a delivery file. **Nothing invokes it at delivery time.** Verified: the only two call sites in the repository are test modules; `monthly_run.sh`, the launchers, and the liveness tools never import it. The rules therefore stop a wrong file being *written*, not a wrong file being *used* — a delivery edited on a branch where the suite is not run, or a rule added after a file was written, is refused by nobody. **This is not new and #428 did not cause it**, but #428 is what makes it worth an entry: until now every rule guarded a property the single-source deliveries could not violate, so the gap cost nothing. The coverage rule is the first one written for a composition that does not yet exist, and the stories that create that composition are the ones listed in the trigger. **Not obviously worth closing by calling `check()` at delivery time** — that would put an import-time refusal on the production path, where the failure mode is a partner getting nothing rather than a developer getting a message, which is the trade ADR-020 §1 is careful about. The cheaper reading is that this is a CI-enforced invariant and should be *described* as one: ADR-019 §4 now says so in as many words (added by #428), so the risk is that someone reads a rule and assumes it protects the run. Cross-refs: **#424**, **#428**, **#429**, **#430**, C-143 (the ADR/code drift these same rules are exposed to), C-125 (why the `targets` stair deliberately stops here). **2026-09-17 (#479):** R1 and R2 now also have a fleet-wide edit-time gate independent of any delivery, `tests/test_ensemble_maturity_rules.py`, mutation-verified. It narrows this entry to the *delivery* rules (tier, reconciliation, coverage); the member-maturity rules are no longer only reachable through `check()`. |

---

### C-145 — A delivery combining sources only for coverage must still declare `reconciled=True`, which nothing verifies

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | **The first delivery whose sources are combined only for target coverage** — every source carrying unique targets, none reconciling with another. #430 (S6) makes such a delivery consumable and #422 decides the cm coverage that would make one realistic. Writing it requires `reconciled=True`, and whoever writes it will either believe the claim or notice it is empty. |
| **Source** | S5 of epic #424 (2026-08-26), verifying rather than assuming per #429 |
| **Status** | Open |
| **Location** | `deliveries/coherence.py:_check_reconciliation` (the `require.reconciled is not True` gate); `docs/ADRs/019_delivery_declaration.md` §4 |
| **Notes** | #429 split the reconciliation requirement from the coverage requirement — every source must now either join the reconciliation group **or** carry targets no other source provides. **The split governs what happens after the `reconciled is not True` gate, not the gate**, which #429 asked to be verified rather than assumed. Verified, with a test (`TestTheSplitDidNotMoveTheGate`): two sources with fully disjoint `provides` and `reconciled=None` still raise the original hard error, and they raise it for the original reason. **The residue is that `reconciled=True` becomes a required incantation for a delivery containing no reconciliation at all.** ADR-019 §4 says the key means the sources are reconciled; a pure-coverage delivery would set it while nothing in the file reconciles with anything, and no rule would object — the split's own exemption is what lets every source through. It is not currently reachable: no delivery in the platform has two sources, and `un_crafd`'s intended three-source shape *does* contain a reconciling pair, so `True` is honest there. **Deliberately not fixed in S5.** Moving the gate is a behaviour change to the `None`/`False` semantics that S2 (#426) documented and pinned four days earlier, and is precisely what #419 proposed and this epic corrected as being a behaviour change rather than a documentation fix. **Two exits, and the choice is the maintainer's:** (a) let `reconciled` mean "reconciled *where the sources overlap*", which makes `True` vacuously honest for disjoint sources and needs only ADR wording; or (b) admit a third state — sources combined for coverage — and let the gate accept it, which needs a new key and a test change. Cross-refs: **#419**, **#424**, **#426**, **#429**, **#430**, **#422**, C-143 (the ADR/code drift these rules are exposed to), C-144 (all of this is edit-time only). |

---

### C-146 — A conformance test that reads config values but never constructs the object is green against exactly the failures that matter

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | **Adding a cross-field constraint to `HydraNetConfig` in views-hydranet without adding a views-models test that CONSTRUCTS each roster config.** The value-comparison suites here cannot see such a constraint by design, so the new rule lands unenforced on this side and the models it rejects go quiet rather than red. |
| **Source** | views-hydranet session (2026-09-07), reported cross-session; deduplicated against C-51 by this register's maintainer |
| **Status** | Open |
| **Location** | `tests/test_roster_conformance.py`; `models/bold_comet/configs/config_hyperparameters.py`, `models/heavy_freighter/configs/config_hyperparameters.py`; fixed by `tests/test_roster_configs_load.py` |
| **Notes** | `bold_comet` and `heavy_freighter` had scheduled sampling active (`ss_schedule='linear'`, `ss_epsilon_max=0.5`) with `ss_feedback` unset. It defaults to `'mean'`, contradicting their own `rollout_feedback='sample'`, so `HydraNetConfig` refused to construct and **neither model could be run at all — from 2026-08-13, through every green CI run.** The ensemble was eight-strong on paper and six in practice. **The validation was never missing.** `HydraNetConfig` raised correctly; views-pipeline-core's `CoreConfigSniffer` is the wrong layer and has no knowledge of `ss_feedback`. What was missing is that **nothing ran that validation over the roster in CI**: `test_roster_conformance.py` compares 184 config values against a reference dict and never constructs a config, so a model can satisfy every pinned value and still be unloadable. **This is the second occurrence of one shape, and that is why it is registered rather than closed with the fix.** C-51 (Resolved 2026-06-01) was the same two-of-four models made unloadable by a missing config key, and its own Notes say why the guard of the day missed it — the parity test *"strips loss keys and compares models pairwise — since all three are equally missing the field, they match each other."* A comparison cannot see a defect its comparands share. The fix then was `test_hydranet_has_sampling_strategy`, a **presence** check for one named key — value inspection again, so it could never have caught this. Three months and two incidents say the class is not closed by adding another value assertion. **The general form: a conformance test that reads values but never constructs the object is green-by-construction against exactly the failures that matter.** It is the same shape as views-hydranet's C-303 (prose asserting a check the code does not implement) one layer out, and the same shape as C-143 here (ADR-019 specifying rules nothing checks against `coherence.py`). **Fixed** by `tests/test_roster_configs_load.py`, which loads all eight and is mutation-verified — reverting either config fix turns it red. **The Tier 1 branch was opened and is now closed on evidence (2026-09-07).** The question was whether a downstream ensemble run proceeded with six members while declaring eight and said nothing, which would be silent output incorrectness. It did not. The timeline settles it: `rusty_bucket`'s last run is **2026-08-04**; the roster swap that put these two models into the ensemble is **2026-08-11** (`feat(#146)`); `bold_comet`'s config last changed 2026-08-10. **The only run on record predates the roster by a week**, against the eight `temporary_*` stand-ins, which loaded. The broken members were never in a pool that executed. Tier 2 therefore stands on evidence rather than on absence of it. Two caveats worth keeping: `prediction_frame_ensemble.py` raises on n_rows / identifier / sample_count mismatch (#160, *"no silent unbalanced pooling"*), so a missing member would probably have failed loud at pooling anyway — but **the actual reason nothing bad happened is that nobody ran it.** The ensemble has been shadow-status since the real models went in, which is a thin thing to have been protected by. **An observation, not a defect:** `violet_visitor` has `ss_epsilon_max=0.0` — scheduled sampling configured but off, while its siblings have 0.5. Treated as intentional by the new test. Declaring it explicitly would be a views-models change, and `violet_visitor` is owned by parallel sessions whose content this repo's convention says not to edit. Cross-refs: **views-hydranet#295**, **C-259** (views-hydranet's own entry), **C-51** (the first occurrence), **C-143**, **C-05**, **C-52**. **2026-09-19 (#501), a third occurrence of the shape:** `test_africa_region` asserted `"africa_me_legacy" in <queryset text>` — a substring anywhere in the file. When the roster moved to `REGION = "land"` the comment recording the history still contained the old name, and the test stayed green on the new region. Replaced by `test_global_land_region`, which reads the assignment line; mutation-verified. The class: a guard that matches a *token* rather than the *statement that decides* is green whenever the token survives in prose. **Cluster G** (added 2026-10-26, #610) — the conformance test reads config values and never constructs the object, so its input excludes every failure that construction would surface. Green against exactly what matters. |

---

### C-147 — Two readers of the same maturity fact disagreed: the catalog copied the §3 map and dropped rule R2

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | **Adding a reader that translates `deployment_status` into a maturity value without calling `deliveries.coherence.maturity_of`.** After #476 the only translating readers are `coherence.py` and the two catalog tools that import it; `run_integration_tests.sh` and `tests/test_roster_configs_load.py` read maturity but only select a file or test for `retired`, which needs no R2. A third translation is the trigger. |
| **Source** | code-review max on PR #476 (agents 1, 2, 3, 4 independently) and review-diff, 2026-09-17 |
| **Status** | Mitigated |
| **Location** | `tools/catalogs/create_catalogs.py`, `tools/catalogs/update_readme.py` (fixed in `de005517`); `deliveries/coherence.py` `maturity_of()` |
| **Notes** | The first commit of #476 gave each catalog tool its own four-entry `_LEGACY_TO_MATURITY` dict, justified in the plan as "WET, deliberately" on the theory that `deliveries/` and `tools/catalogs/` should not import each other. The dict maps `deployed → graduate` unconditionally; ADR-017 §3 R2 says a composite is `graduate` only if every member is. `ensembles/white_mustang` is `deployed` with two `shadow` members, so the README catalog said `graduate` while `maturity_of()` said `candidate` — the same class of silent `deployed → graduate` promotion that got #444 reverted, one layer out. The WET justification was wrong on its own terms: `coherence.py` is stdlib-only, so importing it couples nothing, and the "second incident" that WET-before-DRY waits for had already happened (C-130's table, then this). Fixed by making `maturity_of()` the one translation and having both tools call it. The register keeps the entry because the shape — a rule with a conditional clause, copied as a flat map — is the kind of thing a fourth reader reproduces. Cross-refs: **C-130**, **C-143**, **C-148**, **ADR-017 §3/§11**. |

---

### C-148 — The `deployment_status`-is-inert guard only sees comparisons, so a dict-literal reader is invisible to it

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | **Adding a reader of `deployment_status` (or `maturity`) that maps values through a dict or set literal rather than comparing them with `==`/`!=`.** The guard stays green and the new reader is un-allowlisted. |
| **Source** | code-review max on PR #476, agent 3 (git history), 2026-09-17 |
| **Status** | Open |
| **Location** | `tests/test_deployment_status_inert.py` (`_COMPARISON` regex) |
| **Notes** | The guard's job is to keep the list of code that reads the legacy vocabulary explicit and allowlisted. Its scanner matches `== "shadow"`-style comparisons only. The two `_LEGACY_TO_MATURITY` dicts in the first commit of #476 were new readers of every legacy value, and the guard passed (verified by the reviewing agent: 4 passed with the dicts present). The dicts are gone (C-147), but the blind spot is not: any future value-mapping reader gets the same free pass. This is a guard that cannot fire on one whole shape of the thing it guards. ⚠️ **Citation corrected 2026-10-26 (#610):** this sentence previously cited `reports/measurements/2026-08-23_agent_failure_pattern_prevalence.md` as recording the pattern "in 9 of 18 repositories". **That file does not exist** — not in this repository and nowhere under `views_platform/`, verified 2026-10-26. The figure appears only in the `falsify` skill's own prose (`~/.claude/skills/falsify/SKILL.md:19`), which attributes it to the same missing path. The claim may be true; it is **unverified from here**, and per [ADR-025](../docs/ADRs/025_a_check_that_cannot_see_must_refuse.md) §2.4 an unverifiable witness is not a witness. Kept, attributed, and marked unverified rather than deleted — the prevalence argument is load-bearing for C-159's tier. Not fixed in #476 because the fix is a scanner change with its own mutation test, not a one-liner, and the PR is already carrying the readers. Cross-refs: **C-147**, **C-146** (same "green-by-construction" shape). **Cluster G** (added 2026-10-26, #610) — the guard's scanner sees comparisons only, so a dict-literal reader is outside its input and the guard stays green while the thing it guards is violated. |

---

### C-149 — `run.sh`'s env prefix is whatever the operator types at the scaffold prompt; nothing checks it against `requirements.txt`

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | **Regenerating `run.sh` for an existing model with `tools/scaffold/build_model_scaffold.py`** — the builder asks "Enter the name of the architecture package" and writes the answer straight into `env_path` (`template_run_sh.py:53`); an answer that does not match the engine in `requirements.txt` builds and activates the wrong prefix without a word. |
| **Source** | code-review of #491 (git-history agent), 2026-09-19 |
| **Status** | Open |
| **Location** | `tools/scaffold/build_model_scaffold.py:206-211`; pipeline-core `templates/model/template_run_sh.py:53` |
| **Notes** | On `staging_202608`, commit `4be8f66a` ("tests", 2026-09-08) regenerated `run.sh` for **19 r2darts2 models** and every one came out with `env_path=.../envs/views-hydranet` — the HydraNet prefix — while their `requirements.txt` said `views-r2darts2`. Each was created correctly; the regeneration overwrote them with one wrong prompt answer. Eleven of the 19 reached `development` in #491 with the line corrected; the other eight (`beautiful_people crimson_tide frozen_peak iron_will shadow_wolf swift_current teenage_dirtbag wild_storm`) still carry it on the branch (named on #402). On `development` this cannot land silently: `tests/test_environment_sharing.py` pins the tenant count per prefix and goes red (mutation-verified in #491). On a branch where that suite is not run — staging's case — it does. Tier 3, not 2, because the guard exists here; the exposure is any branch that skips it, and the fix is the builder refusing an answer that contradicts `requirements.txt` (or deriving it from there and not asking). Cross-refs: **C-115** (what a wrong prefix does to 31 models), **C-116**, **#491**, **#402**. |

---

### C-150 — A fresh env resolves PyPI's current torch, whose CUDA build an older driver cannot run; every engine then either crawls on CPU or refuses

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | **Building a model env from `requirements.txt` on a machine whose NVIDIA driver predates the CUDA build of PyPI's current torch** (2026-09-19: torch 2.14 / CUDA 13 against a 535-series driver). `views-hydranet` (`torch>=2.2.1,<3`) and `views-r2darts2` (`darts[torch]`, no bound) both resolve to it. |
| **Source** | views-hydranet#377 (2026-09-16, the 6 h 46 m CPU run); measured again on the laptop for r2darts2 in #485's verification and for HydraNet in #493's; filed as #494 |
| **Status** | Open |
| **Location** | `models/<hydranet>/requirements.txt` (8), `models/<r2darts2>/requirements.txt` (42); `run.sh` via pipeline-core's `template_run_sh.py` |
| **Notes** | Before #493 a HydraNet on such a machine fell back to CPU under a banner and trained at ~2× the GPU time (#377: 6 h 46 m vs 3 h 15 m); r2darts2 hardcodes `accelerator: gpu` and fails at model init. After #493 (`require_cuda: True`, views-hydranet 0.1.1) the HydraNet case is a `RuntimeError` at the top of training — 17 s, zero epochs, measured. Nothing silent remains; what remains is that a fresh env on an older-driver machine is blocked until an operator installs a driver-matching torch by hand (`torch==2.10.0` from the `cu128` index worked here). fimbulthul's driver is newer and unaffected this week. Whose fix it is — an engine ceiling, a `run.sh` index step, or an operator runbook line — is #494's question. Cross-refs: **#494**, **#493**, **#485**, **views-hydranet#377**, **C-116**. |

---

### C-151 — viewser's `toolz<0.12` pin resolves to a toolz that cannot import `tlz` submodules on current Python 3.11; any reinstall of viewser into a datafactory env re-breaks every datafactory fetch

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | **Running any model's `run.sh` — or `pip install -r requirements.txt` for any viewser tenant — inside an env that also serves datafactory models** (fimbulthul's `views_pipeline`). `run.sh`'s dry-run check reads a violated pin as "outdated" and reinstalls, and viewser 6.6.4 / views-partitioning 3.0.1 pin `toolz<0.12.0`, so toolz goes back to 0.11.2. |
| **Source** | fimbulthul, 2026-09-19: the eleven-model calibration pass (#499) failed 4 × in 9 s at data fetch — `AttributeError: 'TlzSpec' object has no attribute '_uninitialized_submodules'`; diagnosed with the views-baseline session |
| **Status** | Open |
| **Location** | fimbulthul `views_pipeline` env; `models/*/run.sh` (the dry-run reinstall, pipeline-core's `template_run_sh.py`); viewser 6.6.4 and views-partitioning 3.0.1 metadata (not ours) |
| **Notes** | `datafactory_query` imports dask, dask imports `tlz` (toolz's lazy shim), and toolz 0.11.2's `TlzSpec` predates the `_uninitialized_submodules` attribute current CPython 3.11.x importlib requires on submodule import — `import tlz` succeeds, `import tlz.curried` dies. toolz ≥0.12.1 fixes it (reproduced by views-baseline on 3.11.13/3.11.14/3.11.15; fixed with 0.12.1 and 1.1.0). But every *correct* resolution of viewser's pin lands on the broken 0.11.2, so the working state is a pin violation kept alive by hand: Simon ran `pip install "toolz>=0.12.1"` at 12:05, something reinstalled viewser's pins in the afternoon (a `run.sh`, most likely), and at 21:29 the pass died again; re-upgraded 21:45. **Rule for that env until fixed:** the integration runner only (it runs `main.py` directly); never a model's `run.sh` in `views_pipeline`. The fix is upstream — viewser and views-partitioning lifting `toolz<0.12` (both orphaned, #473) — or the two worlds in two envs (C-116). Tier 2: it recurs on a routine action and takes every datafactory model with it, loud but at 9 s per model into a multi-hour pass. Cross-refs: **C-116**, **#473**, **#499**, **#494** (the other fresh-env trap). |

---

### C-152 — `tools/collapse` is a third reader of views-hydranet's on-disk prediction layout, which is not a published contract

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | **views-hydranet changing where or how `PredictionFrame` artifacts land on disk** — renaming `predictions_<run_type>_<ts>/origin_i/<target>/`, the `y_pred.npy` / `identifiers.npz` filenames, or the `time` / `unit` keys inside the identifiers archive. Also: adding a target directory alongside `lr_*` / `by_*` that the converter's fixed `TARGETS` would silently not read. |
| **Source** | ADR-023 §Consequences/Negative, written with the converter (2026-09-28, #505) — declared by the author, not found by an audit |
| **Status** | Open |
| **Location** | `tools/collapse/collapse_predictions.py` (`_load_target`, `convert_model`); the producer is `views-hydranet@32bc509` `views_hydranet/utils/prediction_frame_assembler.py` and the pipeline-core writer |
| **Notes** | views-hydranet writes the layout, pipeline-core reads it back, and views-models now parses it too — but no ADR or schema in any of the three repos states it, so a rename in the producing repo is not visibly a breaking change to anyone. **The mitigation is loudness, not prevention:** every structural assumption in the converter raises `CollapseError` naming the offending path, and `tests/test_collapse_on_real_predictions.py` runs over whatever real output the machine holds, so a layout change fails on the operator's next run rather than producing a short or mislabelled parquet. Tier 3 and not 2 because the failure is immediate and legible, and the converter is invoked by hand under supervision rather than inside an automated delivery. It becomes Tier 2 the moment anything schedules it. The right long-term fix is for the layout to be declared once in views-hydranet and imported, not re-described — the same shape as **C-133** / `vmo_021` (a derivation, not three readers agreeing by luck). Cross-refs: **C-47** (why the pipeline's own parquet is off and this converter exists), **#505**, ADR-023. |

---

### C-153 — the datafactory credential crosses the public internet readable, and we have now put it on hardware we do not own

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | **Running any model on rented, third-party, or otherwise untrusted hardware** — a cloud GPU, a collaborator's machine, a CI runner outside our control. Also: leaving a stopped pod, a snapshot, or a detached volume in existence after a campaign ends. |
| **Source** | First RunPod deployment, 2026-09-28 (#499, #508); raised by the views-datafactory session reviewing `reports/postmortem_runpod_first_deployment_2026-09.md` |
| **Status** | Accepted |
| **Location** | `~/.netrc` on any rented machine; `datafactory_query.defaults.RemoteConfig(server="…", scheme="http")`; views-datafactory register **C-318** is the same fact from the producing side |
| **Notes** | The datafactory speaks **plain HTTP**. HTTP Basic sends the credential base64-encoded on every chunk request — base64 is encoding, not encryption — so the password is readable by anything on the path. views-datafactory accepted this (**their C-318**) when the audience was a trusted circle on trusted networks, and on fimbulthul that was reasonable. **We changed the audience without changing the mechanism:** on 2026-09-28 the credential was placed on five rented machines in datacentres we do not control, and the operator chose knowingly to use his personal login rather than provision a throwaway. That choice is recorded, not second-guessed — the work was owed and the server was gone. Two properties make the residual risk outlive the run: the credential has **no expiry** and **no per-host registration**, so it stays valid until a person rotates it by hand, and it authenticates from anywhere. A pod image, a snapshot or a volume that outlives a campaign therefore carries a live, permanent credential. **Tier 3 and not 2** because the exposure is a real but unquantified interception risk rather than a demonstrated compromise, and because the mitigations are cheap and known. **Mitigations, in order of preference:** a throwaway login for the campaign, retired after (about three commands for whoever administers the data server); TLS on the data server, which removes the class; or, failing both, deleting rented volumes and images at campaign end and rotating afterwards. `docs/runpod_run_guide.md` states the throwaway option and tells the operator it is cheaper than it looks. Cross-refs: **C-151** (the other thing that bites a fresh datafactory environment), views-datafactory **C-318**, views-models **#509** (the client floor still permits a version that leaked the credential across redirects). |

---

### C-154 — `/workspace` on RunPod silently ignores `chmod`, so a credential placed there stays world-readable with no error

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Placing any credential on rented hardware — or writing a runbook step that says `chmod 600` without saying which filesystem it must be on. |
| **Source** | Found 2026-09-29 while placing the Appwrite publish credentials during the first RunPod deployment; #518 |
| **Status** | Mitigated |
| **Location** | `docs/runpod_run_guide.md` Step 2.3 (the warning, and the `umask 077` that actually protects the file); `tools/podrun/pod_run_model.sh` and `tools/podrun/pod_run_darts_calibration.sh` (both `stat -c %a /root/.netrc` rather than trusting their own `chmod`) |
| **Notes** | **Written 2026-10-07 because the register did not contain it.** The guide has cited "views-models C-154" since 2026-09-29 (`docs/runpod_run_guide.md:192`) and no such entry existed — a citation resolving to nothing, which is the documentation equivalent of a guard that cannot fire: it stops the next reader looking. The facts are the guide's own, not new here. `/workspace` is a network filesystem; `chmod 600` there **returns success and does nothing**, leaving the file mode `666` and readable by every process on the machine, with no error to notice. `stat -c %a` is the only way to find out, and only if you think to look. The Appwrite publish credentials sat world-readable on rented hardware until they were moved to `/root/.secrets`. **Mitigations:** credentials go on `/root` (local disk) and never `/workspace`; the guide's placement command uses `umask 077`, which is what actually protects the file in transit; both pod runners verify the mode with `stat` instead of trusting a `chmod` they issued. **Residual:** the protection is a convention plus two scripts that check one specific path. Nothing prevents a future runner, or an operator following a different instruction, from writing a secret to `/workspace` — and it will look like it worked. Cross-refs: **C-153** (the datafactory credential is now on hardware we do not own), **C-155** (also found by measuring rather than by a failure). |

---

### C-155 — The `dataframe` prediction tier materialises every rolling origin as Python lists, so a sample count that passes every check cannot run on any machine we can rent

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Raising `num_samples` on any pgm model whose `prediction_format` is `"dataframe"` — or #492 landing and someone restoring the values it makes affordable, without re-measuring. |
| **Source** | Measured during epic #532 (2026-10-06/07), while establishing why #536's two models could not be run at all |
| **Status** | Mitigated |
| **Location** | `views_r2darts2/transformers/darts_bridge.py::prediction_frames_to_dataframe` (a Python list per cell); `views_r2darts2/engines/darts_forecasting_model_manager.py:353-362` (all origins held before any release). Consumers: `models/{little_talks,mister_bluesky}/configs/config_hyperparameters.py`. |
| **Notes** | **Nothing in the configuration says this is impossible, and every check passes.** The engine converts each prediction to a list-in-cell DataFrame, and the evaluation path builds **all 13 rolling origins before releasing any of them** — `results` is a list and `_release_scratch_if_frames_copied` runs once after the loop — so peak memory is the whole evaluation, not one origin. Measured at pgm (3 targets x 2,333,448 rows x 13 origins, distinct float objects): `num_samples` 1 -> ~11 GB, 4 -> ~18 GB, 8 -> ~29 GB, 16 -> ~52 GB, **100 -> ~303 GB**, against a pod selection rule of RAM >= 50 GB (`docs/runpod_run_guide.md` Phase 1.1). `little_talks` and `mister_bluesky` sat at 100 and were therefore unrunnable on any hardware available to this project — discovered by arithmetic, not by a failure, because nothing had tried. **Why Mitigated and not Open:** #536/#542 lowered both to 1 and added `tests/test_sample_count_matches_declared_metrics.py`; #534's pod preflight refuses `num_samples != 1` by name before any GPU time. Both are partial — the test guards the *metric pairing*, not the magnitude, and the preflight guards only the pod path, so a local run or a config edit can still produce an unrunnable model. **The class fix is #492** (migrate to `prediction_format: "prediction_frame"`, which hands memmaps instead of Python lists); both config files name it as their revert trigger, and the measurement is recorded on that issue. *Carry forward when #492 lands:* the memmap path is cheaper per unit but has the **same shape** — `PredictionScratch` instances also accumulate in a list and are freed only after the loop (views-r2darts2#54), so disk peak is still `N_origins x scratch`. Re-measure before restoring any sample count. Cross-refs: **C-116** (shared environments), **C-152** (a third reader of an unpublished on-disk layout). |

---

### C-156 — Every config check that loads via `spec_from_file_location` can read stale `__pycache__` bytecode, so a same-length config edit is invisible to it

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Any config edit that does not change the file's byte length — `"num_samples": 1,` -> `"num_samples": 5,`, `300` -> `400` — followed by a check that loads that config. Most acute where a script edits a config and then re-reads it to verify the edit. |
| **Source** | Found 2026-10-07 while mutation-testing #536's new guard; the mutation's "restoration" was verified by sha256 and the next run still read the mutated value |
| **Status** | Open |
| **Location** | 13 test files load configs this way (incl. `tests/conftest.py`, `test_roster_configs_load.py`, `test_darts_entity_id_matches_level.py`, `test_roster_conformance.py`); `tools/catalogs/create_catalogs.py`; and the embedded config checks in `tools/podrun/pod_run_model.sh` and `tools/podrun/pod_run_darts_calibration.sh`. |
| **Notes** | **`exec_module` reuses `__pycache__` when the cached bytecode's recorded source size and mtime still match.** Size is the weak half: config edits routinely preserve it exactly. Demonstrated — mutating `dark_river`'s `num_samples` 1 -> 5 (same length), restoring the source, and confirming byte-identity by sha256 still left the test reading **5**, because the stale `.pyc` was reused. The source was correct and the check was reading something else. Clearing `models/**/__pycache__` made it green. **Why this is worse than a flaky test:** the failure direction is unbounded. A guard can pass on a config it is not reading (false green on a real defect) or fail on one it is not reading (a day spent on a correct file, which is what happened here). **Worst instance is not in the tests.** `pod_run_model.sh`'s `--rehearsal` patches the pod's config and then re-imports it to *verify the patch took* — its own comment says "VERIFY by re-importing, not by trusting the substitution". `importlib.invalidate_caches()` there clears finder caches, **not** bytecode staleness. Today it is safe only by luck: `'total_lessons': 300` -> `40` changes the length. A same-length patch (`300` -> `400`) would report a verified rehearsal while the model trained a different budget — the precise failure the verification exists to prevent. **Mitigated in one place only:** `tests/test_sample_count_matches_declared_metrics.py::_config` compiles from source text instead, with the reasoning in its docstring, and is proven immune (mutate, poison the cache, restore, still green). Not propagated to the other 15 sites in #542, which is scoped to #536. **Fixes, cheapest first:** (a) `sys.dont_write_bytecode = True` in `tests/conftest.py` and in the podrun heredocs — stops new caches but does not ignore existing ones; (b) compile from source at each load site, as the mitigated test does; (c) `PYTHONDONTWRITEBYTECODE=1` in CI and in the pod runners, which fixes CI and pods but not a developer's laptop. Cross-refs: **C-155** (also found by measuring rather than by a failure), **#501** (the guard that was not one — same class: a check that cannot see what it claims to). Member of **Cluster H** (no identity separates current from left-over). |

---

### C-157 — Model config fields have no schema: a dead key, a missing key and a look-alike key all pass every check and are indistinguishable by reading

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Adding, renaming or removing a key in any `configs/config_*.py` — or cloning a model's config to make a new one, which is how the roster was built. |
| **Source** | Four instances found in four days during epic #532 (2026-10-06 to 2026-10-09), each by a run failing rather than by a check. **A fifth found 2026-10-24 by `/falsify` — the first found by a check rather than by a failure.** |
| **Status** | Open |
| **Location** | `models/*/configs/config_hyperparameters.py` and `config_meta.py` (129 models); consumed by views-r2darts2, **views-hydranet** and views-pipeline-core, none of which validates the set of keys it is given. views-hydranet's `HydraNetConfig` is explicitly `extra="allow"` (`views_hydranet/utils/config_initializer.py`), so an unknown key is accepted by construction. |
| **Notes** | **Nothing anywhere states which keys exist, which are required, which are consumed, or which must agree with each other.** Each config is a dict that is passed through several repositories and read by `.get()`. A key that is misspelt, dead, missing or merely *adjacent* to the one that matters all behave identically: silently. Four instances, each costing a run or a model: **(1) MISSING — `brave_heart` had no `target_scaler`. views-r2darts2 scales the target through that key alone (`dataset/base.py:1192-1223`); without it the loss saw raw counts reaching 113,395 and `logcosh` overflowed float32, dying with `per_channel=[nan, nan, nan]` after ~90 GPU-minutes. Its 39 darts siblings all declare it. Fixed #537; guarded by `tests/test_target_scaler_is_declared_with_a_loss.py`. **(2) LOOK-ALIKE — the same config carries a `feature_scaler_map` that explicitly lists `lr_ged_sb/ns/os` under `AsinhTransform`. It reads exactly like target scaling and is not: that map applies to columns used as features. This is presumably how the omission survived review. **(3) DEAD — `force_target_only` has zero references in views-r2darts2 0.2.4. Nine models set it; some `True`, some `False`; it changes nothing. A reader cannot tell it from a live key. **(4) INCONSISTENT SET — `num_samples`, `mc_dropout` and `regression_point_metrics`/`regression_sample_metrics` must agree or the run trains to completion and then raises "No metrics configured for (regression, point)". Nothing checked that until #536, and the same change collapsed two models onto siblings because those keys were the *only* thing distinguishing them (C-158). ⛔ **(5) DEAD, SECOND INSTANCE, DIFFERENT REPOSITORY — `h_init`, found 2026-10-24 by `/falsify` on the claim that `heavy_freighter` needs no further change in either repository.** 15 HydraNet sources declared `'h_init': 'abs_rand_exp-100'`. It is read by **nothing**: zero hits for `h_init` and zero for `abs_rand_exp` across the package source of views-hydranet **0.1.5** (installed) *and* **0.1.7** (current PyPI, wheel downloaded and inspected), and the hidden state is initialised unconditionally — `architectures/HydraBNrecurrentUnet_06_LSTM4.py:765 init_hTtime()` returns `torch.zeros((1, hidden_channels, H, W))` with the docstring *"Zero-initialized state"*. **No result ever depended on the value, so nothing already produced is wrong** — the harm is that the config asserts an initialisation scheme the library has never had, and a reader cannot tell it from a live key. This is instance (3)'s shape exactly, one repository along, which is the argument for a schema rather than a fifth bespoke guard. **Deleted from 7 sources 2026-10-24** (`blazing_meteor`, `blue_stranger`, `bold_comet`, `bright_starship`, `heavy_freighter`, `pink_pirate`, `purple_alien`); all 7 still load and the full roster suite stays green. ✅ **The remaining 8 were DELETED 2026-10-27 (#630, PR #632)** — one atomic commit across all eight, which is exactly the shape this entry said was required. They had been recorded as BLOCKED, not overlooked: `amber_kestrel`, `copper_kestrel`, `silver_kestrel`, `bronze_kestrel`, `cobalt_kestrel`, `indigo_kestrel`, `russet_kestrel`, `violet_visitor`. They form a pinned one-variable contrast chain (`violet_visitor` ← `amber_kestrel` ← `copper_kestrel`; `amber_kestrel` ← `silver`/`bronze`/`cobalt` as seed replicates — `tests/test_roster_conformance.py:121` `EXPERIMENT_ARMS`, `:176` `SEED_REPLICATES`), `h_init` is in neither `ARM_FREE_KEYS` nor `REPLICATE_FREE_KEYS`, so removing it from an arm but not its control breaks the contrast and fires the paired guards. `amber_kestrel`'s control is the 2026-10-12 `violet_visitor` artefact, which **cannot be retrained** (views-hydranet C-317), and `violet_visitor` is owned by a parallel session. **The fix for those 8 was one atomic edit across all 8 including `violet_visitor`, or none** — a partial deletion is worse than leaving it. ⛔ **What unblocked it was not a technical change.** The block was the standing rule never to edit `violet_visitor`'s content, whose stated reason is that a parallel session owns it; Simon confirmed he owns all eight and lifted it for this edit. Recorded because a rule relaxed silently is a rule that is gone. `ARM_FREE_KEYS`/`REPLICATE_FREE_KEYS` govern what a pair may **differ** on, so removing the key from all eight changes no difference between any pair and every one-variable contrast survives — `tests/test_roster_conformance.py` 297 passed. Red-by-design guard at `tests/test_falsification_heavy_freighter_launch_readiness.py`, which names the remaining sources and reported 8 (down from 15); **it now reports 0.** **Why a schema and not more tests:** each of the four was caught by writing one more bespoke guard after one more failed run. That scales linearly with keys and finds nothing in advance. The asymmetry is the argument — a key that is read is exercised by every run, while a key that is *not* read is exercised by nothing, so the cheap win is enumerating the keys each consumer actually reads and refusing the rest. **Not actioned here** (#537 fixes one instance); filed as views-models#546 with the four as evidence. Cross-refs: **C-95** (`feature_scaler_map` silently skips unmapped features — the same family), **C-156** (a check that cannot see what it claims to), **C-158**. **Cluster G** (added 2026-10-26, #610) — ⛔ **the structural root of the config-side members.** A dead key, a missing key and a look-alike key are indistinguishable *by reading*, which is the same statement as: the check's input cannot discriminate the thing it is asked about. C-175 is this entry restated as a procedural failure, and C-165/C-166/C-167 are its instances. |

---

### C-158 — Two models were distinguished from siblings only by sample-count keys, so normalising those keys silently made them duplicates

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Changing `num_samples` or `mc_dropout` on any model — or any edit intended to make models "consistent" with each other. |
| **Source** | Measured 2026-10-08 on real predictions after #536 shipped |
| **Status** | Open |
| **Location** | `models/mister_bluesky` (= `dancing_monkey`), `models/little_talks` (= `dark_necessities`) |
| **Notes** | #536 set `num_samples: 100 -> 1` and `mc_dropout: True -> False` on two models so they could run at all (the 100-sample path needs ~303 GB, C-155). Those two keys were the **entire** functional difference between each and an existing sibling: same architecture, same queryset, same hyperparameters, same seed — verified by diff, and then by the output, which is **byte-identical** (`corr = 1.0000`, identical sums and non-zero fractions, produced on two different pods). They were never independent models; they were the probabilistic variants of `dancing_monkey` and `dark_necessities`, and removing the probabilistic part removed the model. **Consequence beyond the count:** the consumer is ensemble-diversity work, where shipping one model twice double-weights an architecture and corrupts the measurement — worse than shipping nine instead of eleven. **What did not catch it:** the guard added in #536 asserted the delivery batch was internally *comparable*, and identical is maximally comparable. A test can be true and measure the wrong property. **The fix is not to re-split them by hand** but views-models#492, which removes the memory ceiling so they can run at 100 samples and be themselves again. Until then the delivery is 8 distinct models, and saying "eleven files" would be false. Cross-refs: **C-155**, **C-157**. |

---

### C-159 — A mutation that silently fails to apply reads as a RESULT, so the procedure we use to validate guards can report either kind of false answer

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Running a guard audit (`falsify guard`, `ship-it` Step 5.5, or any by-hand mutation pass) where a mutation is applied by string replacement, `sed`, or a patch script rather than by an edit whose effect is read back. |
| **Source** | Observed twice in one session, 2026-10-12, during the #549/#552 guard audits |
| **Status** | Open |
| **Location** | The audit procedure itself, not a file. Nearest code: `~/.claude/skills/falsify/references/guard_mutations.md` (normative for guard mode, requires every mutation be *applied and executed*, does not require its application be *verified*). |
| **Notes** | Mutation testing is the only instrument we have for the question "can this guard fire?", and the repository relies on it heavily — guards that cannot fire were reportedly recorded in **9 of 18 repositories** — ⚠️ a figure stated only in the `falsify` skill's prose (`~/.claude/skills/falsify/SKILL.md:19`), which attributes it to `reports/measurements/2026-08-23_agent_failure_pattern_prevalence.md`; **that file does not exist anywhere under `views_platform/`** (verified 2026-10-26, #610), so the figure is **unverified from here** and is not used as evidence. That instrument has no check on its own input. A mutation is a textual edit; if the edit does not land, or lands somewhere else, the test run still produces a clean pass/fail that is then recorded as a verdict about the guard. **Both directions are wrong, and they are not equally dangerous. (1) FALSE SURVIVOR — the mutation does not apply, the suite passes, and that is recorded as "the guard did not fire".** Measured instance: injecting the **views-hydranet C-259** defect (scheduled-sampling exposure decoupled by two config keys — verified 2026-10-26 as `views-hydranet/reports/technical_risk_register.md:2027`) into `models/amber_kestrel/configs/config_hyperparameters.py` by inserting `'ss_epsilon_max': 0.5` *earlier* in the same dict literal. Python takes the last key, the file already declared `'ss_epsilon_max': 0.0` further down, so the arm was unchanged. Reported as a survivor until the config was re-read. Cost is wasted hardening of something already sound — annoying, self-limiting. **(2) FALSE CAUGHT — the mutation applies but breaks something unrelated, the suite fails for that reason, and that is recorded as "the guard holds".** This is the one that makes an audit lie, and it needs no exotic mistake: a `sed` that mangles quoting, or an edit that leaves a shell script failing `bash -n`, turns every test in the file red. A decorative guard then passes its audit, and the repository starts believing in protection it does not have — which is precisely the belief `ship-it` Step 5.5 exists to prevent forming. **Why it is not merely operator error:** the same session also had four separate instances of an *assertion* matching prose about the code instead of the code (a comment containing `-t -e`; a paragraph saying "this phase has no rsync step"; a `die` message naming `main.py`; a comment in `requirements.txt` quoting `>=1.9.0`). The common root is a textual operation whose effect on the thing that executes was never read back. Mutation application is the same operation performed by the auditor, with the verdict as its output. **The cheap fix is a precondition, not a tool:** after applying a mutation, assert that the subject changed in the intended direction and in no other — re-read the value through the loader the code uses, or diff the file and check the diff is the one intended — and record `UNRUN` when it did not. A syntax check on the mutated artefact (`bash -n`, an import) separates "the guard fired" from "I broke the file". **Not fixed here.** Both instances above were caught by re-reading, which is not a mechanism. Cross-refs: **C-156** (a check that cannot see what it claims to — the same failure one layer down, in the loader rather than in the auditor), **C-61** (a regex config edit that matched a *commented* line — the same textual-edit-not-read-back root, in tooling rather than in an audit), **C-157** (a key that is misspelt, dead or merely adjacent behaves identically: silently). Member of **Cluster G** (a check out of scope returns a clean verdict). |

---

### C-160 — A cross-repo citation pinned to a commit is stable but unverified, and pinning reads as having checked it

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Replacing a bare `other_repo/file.py:<line>` reference with a pinned GitHub permalink, or adding a new pinned permalink to another repository, in a comment, docstring, ADR or test rationale. |
| **Source** | review-diff, during the merge ritual for PR #557 (2026-10-13) |
| **Status** | Open |
| **Location** | `models/amber_kestrel/configs/config_hyperparameters.py:90-105` (the instance); the convention itself is `CLAUDE.md` ("Cross-repo contracts ... are referenced **by URL at a pinned commit, never copied**") |
| **Notes** | The convention is right and should stay: a bare `file:line` into another repository silently rots the moment that repository is edited, and `reports/delivery_notes/README.md:204-234` already records what a dead pinned link costs a recipient. But pinning solves **drift**, not **accuracy**, and the two feel like the same thing while you are doing the work. PR #557 existed *specifically* to correct two false claims about views-hydranet's ADR-069 rules, and the corrected version shipped a third: the arm's comment said `truncated_nb` + `soft_gate` is admitted by "composition rule (2)", anchored at `config_initializer.py#L1145-L1152`. Rule (2) is the engine's own numbering for that range, so the number and the range agreed with each other — and both were beside the point, because rule (2) *refuses* a non-self-zeroed family that asks for `self_zeroed` composition. What admits this arm is rule (1) (`is_self_zeroed_family and comp in _GATE_COMPOSITIONS`, L1132-L1141) **not firing**. The same hunk asserted `SELF_ZEROED_FAMILIES == {"zinb"}` inline beside a `config_initializer` URL; the set is `frozenset({"zinb"})` in `distributions/registry.py:51`, reached through `self_zeroed_family_names()`. **Why the pin made it worse rather than better:** a reviewer seeing a 40-character sha and an explicit line range reads diligence, and a reader who follows the link lands on real code that really does raise a `ValueError` about composition — plausible enough to stop reading. The wrong anchor is harder to catch than the drifting one it replaced, because a drifted `:1176` eventually points at nothing recognisable while a wrong permalink points at something forever. **What makes this register-worthy rather than a typo:** the comment in question is an *instruction* — "Do not add one back to match violet_visitor" — and its authority is the citation. A future contributor who checks the instruction against the wrong rule can reasonably conclude the instruction is wrong. **No mechanism checks any of this.** `roster_configs_load.yml` installs `views-hydranet==0.1.1` and would catch a config that does not *load*; nothing compares a comment's claim against the lines it points at, and nothing fails when a pinned sha is not the version CI installs. **The cheap discipline, applied here:** before citing, `git show <sha>:<path> | sed -n '<range>p'` and read it; confirm the sha resolves to the version CI actually installs (`git rev-parse v0.1.1^{commit}`); and cite a symbol where it is **defined**, not where it is consumed. All three were done for this fix — e3a039f7 is exactly `v0.1.1^{commit}`, CI installs 0.1.1 at `roster_configs_load.yml:53`, and the three cited ranges are byte-identical at tag and commit. **Not mechanised.** A guard could assert that every pinned views-hydranet sha in `models/**` equals the CI-installed tag, which would catch staleness but still not accuracy. Cross-refs: **C-157** (a dead, missing or look-alike config key is indistinguishable by reading — the same "looks right, is inert" shape in configuration rather than in citation), **C-152** (`tools/collapse` reads views-hydranet's on-disk layout, which is not a published contract — the same cross-repo coupling that these citations are standing in for), **C-159** (a textual operation whose effect on the executing thing was never read back; here the operation is a citation and the unread thing is the other repo's source). Member of **Cluster G** (a check out of scope returns a clean verdict). |

---

### C-161 — The pod runner archives half the posterior and reports `OK`, and the guard that exists to prevent exactly that counts only the half it archives

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Destroying a rented pod after `pod_run_model.sh` reports `STATUS: OK`, for any model whose readout depends on the gate rather than the magnitude body — or re-reading `MANIFEST`'s `raw draws:` / `compressed:` pair as a compression ratio. |
| **Source** | Observed 2026-10-13 retrieving the `amber_kestrel` calibration run, one re-check before the pod was to be destroyed (views-models#558) |
| **Status** | Open |
| **Location** | `tools/podrun/pod_run_model.sh:625-645` (`stage compress_draws`, the `-path '*/lr_*'` filter and the `N_DRAWS` guard); `:652` (`MANIFEST`'s `raw draws:`); `docs/runpod_run_guide.md` Phase 5 ("the full posterior") |
| **Notes** | HydraNet writes six target directories per origin — `lr_{sb,ns,os}_best` (the magnitude body) and `by_{sb,ns,os}_best` (the gate). The archive step selects `-path '*/lr_*'`, so it ships three of six: measured on this run, **6.2 GB of 13 GB archived, 6.2 GB left on `/workspace` to die with the pod**. The header comment is honest ("the `lr_*` posterior"); nothing a reader actually looks at is. `MANIFEST` prints `raw draws: 13G` next to `compressed: 34M` as though one were the compression of the other (the 2026-09-28 `violet_visitor` MANIFEST reads `13G` / `8.4M` the same way), and the run guide calls `draws/` "**the full posterior** ... ~236x smaller than raw" — a ratio whose denominator includes the half that was thrown away. **The guard is the interesting part.** `N_DRAWS`/`N_ARCHIVED` was added specifically so an empty archive could not pass as success; its own comment says "the only other signal would be a small number in MANIFEST that a human has to notice. That is the failure this block exists to make impossible." It counts `lr_*`, archives `lr_*`, and compares the two — so it certifies the subset it was written for and is structurally incapable of noticing the other half. A guard that defines its own scope cannot detect that the scope is wrong. **What it cost:** `amber_kestrel` is the truncated-body / `soft_gate` arm, where emitted P(y>0) *equals the gate exactly* — the gate is the arm — and its config carries a pre-registered readout naming **gate AP as a one-sided kill**. No metricframe holds AP: the gate's only metric is `Brier_cls_sample`, verified on both the arm and the control (`metricframe_by_*` = `['Brier_cls_sample']`, `metricframe_lr_*` = `['CRPS','MCR_sample','QS_sample']`). So AP is computable only from the gate posterior that this script discards. `find models -name "*_by.tar.zst"` returns nothing: **no model in this repository has its gate posterior**, and `violet_visitor`'s 2026-10-12 control is past recovery — weights retained, pod gone. The arm's copy survives only because it was archived by hand on the pod before clearing. Had the pod been destroyed on the first "safe to clear" assessment, the run would have kept its weights, 13 parquets and regression scores and permanently lost the measurement it existed to make. **Not fixed; `pod_run_model.sh` is production and #558 records the options** (archive all six; or two archives `_lr`/`_by`; plus making the omission visible in `MANIFEST` and the guide). Whichever is taken, the guard must count the same set it archives. ⛔ **2026-10-25 — SECOND INSTANCE, on the control run, and the trigger fired verbatim.** `heavy_freighter`'s 300-lesson control run reported `STATUS: OK`, its bundle was verified file-by-file against the pod (**2754/2754 sha256 matches**), and the pod was declared ready to destroy — twice. `/falsify` then found the gate posterior still on `/workspace`: **6.55 GB raw, 632 MB zstd, 78 files, 13 origins × 3 targets**, rescued with the pod minutes from termination and now at `~/views-diagnostics/hf_control_gate_draws/` (78/78 verified by per-member content hash). **This entry's Trigger describes exactly what happened and it was not consulted.** That is the finding worth more than the bytes: the register already held the answer. See **C-168**, which is the root. ⛔ **Two measurements this instance adds that the 2026-10-13 one could not.** (1) **The `MANIFEST` ratio is not merely optimistic, it is arithmetically impossible for the set it appears to describe.** `lr_*` compresses **1256:1** (149,340,800 B → 118,849 B, zero-heavy counts); `by_*` compresses **8.9:1** (→ 16,727,744 B, probabilities in [0,1]). A reader seeing `raw draws: 13G` / `compressed: 9.3M` infers ~1400:1, which no six-family set can reach — so the pair is self-refuting to anyone who checks, and invisible to everyone who does not. (2) **The gate is absent from the parquets as well as the archive.** All 13 delivered parquets carry exactly `month_id, priogrid_id, pred_lr_sb_best, pred_lr_ns_best, pred_lr_os_best` — 30,334,824 rows, no `by_*` column anywhere — because `tools/collapse/collapse_predictions.py:46-48` sets `TARGETS = ("lr_sb_best", "lr_ns_best", "lr_os_best")` with the comment *"`by_*` are gate probabilities, not fatalities, and are deliberately absent."* **So two independent places agree that the gate does not ship, and because they agree, neither can cross-check the other.** Both are honest in their own terms; the defect is that nothing reconciles them against what a gate-calibration run needs. ⚠️ **"No model in this repository has its gate posterior" is now false, and only by hand:** `heavy_freighter`'s control has one because `/falsify` caught it, off-repo, by a one-off stream. Nothing in the runner changed. Cross-refs: **C-168** (ROOT — completeness verified against the producer's own declared scope; this entry is its sharpest instance and its sharpest indictment), **C-170** (four further classes of run output outside `$OUT`), **#549** (the run reported `OK` and the weights died with the pod — same class, different artefact), **views-hydranet C-317** (not bit-reproducible, so a lost posterior needs a new GPU run), **C-94**/**C-155** (an operation reports success having written nothing), **C-159**/**C-160** (a check that cannot read its input reports a clean answer), **C-153**/**C-154** (the other things rented hardware takes with it). Member of **Cluster G** (a check out of scope returns a clean verdict). ⛔ **2026-10-26 — THIRD INSTANCE, and it is no longer about the bytes.** On the `heavy_freighter` #607 structure-penalty arm run, `docs/runpod_run_guide.md` Phase 5 now carries an explicit remedy this entry predates: *"If your readout is gate-side, you must slice the gate before you destroy the pod"*, naming `tools/podrun/slice_gate_posterior.py` and requiring it **on the pod, after `OK`, before the rsync**. **It was not run.** A raw `by_*` tarball was streamed instead — 78 members, 6.10 GB raw / 683 MB zstd, a strict superset of the h=1 slice — and that substitution was reasoned about as sufficient because it contained more data. ✅ **The lesson this instance adds, which the first two could not: `slice_gate_posterior.py` is not only the retriever, it is the VALIDATOR.** It refuses on no `by_*` directories, a target missing from an origin, a row count that is not 64,818, a draw count that disagrees between targets, a non-finite probability, or an archive whose entry count does not match what was written. Holding a superset of the bytes runs **none** of those checks. **A superset of the data is not a substitute for the instrument that judges the data**, because the judgement is precisely what would have revealed the bytes to be wrong — and treating "I have more than it would have given me" as equivalent is the ad hoc rescue that makes a falsification pass worthless. Run afterwards on the still-live pod it exited **0**: 13 origins × 3 targets, every row count exactly 64,818, 16 draws consistent across targets, all probabilities finite in (0,1); `gate_h1.tar.zst` (43,783,836 B) verified locally with the tool's own `gate_h1.sha256`. ⚠️ **So the guide's remedy works and was still skipped**, which moves this entry's open question from *"how do we get the gate off the pod"* to *"what makes a mandatory teardown step in a runbook actually execute"* — the same question #551 asks for the FAO artefacts and #602 asks for the config pre-flight. Cross-ref **C-179**, **C-180** (the other two falsifications from the same audit). |

---

### C-162 — The multitask balancer's learning rate is never updated, and the config default is the path on which that happens

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Writing a new HydraNet config that omits `freeze_multitask_balancer`, or setting it to `False` in a config or a sweep in order to "let the balancer learn" |
| **Source** | views-hydranet method review 2026-10-15 (M-1), relayed and verified against this repository's configs on 2026-10-16 |
| **Location** | `tests/test_config_completeness.py::TestConfigHyperparameters::test_hydranet_declares_the_balancer_frozen` (the guard); the 10 HydraNet `config_hyperparameters.py` files that declare the key; mechanism in views-hydranet `training_engine.py:134-136` and `choose_scheduler` |
| **Status** | Mitigated |
| **Notes** | **The unfrozen path cannot do what unfreezing is for, and says nothing about it.** `choose_scheduler` builds the WarmupDecay scheduler over an `AdamW` constructed from `unet.parameters()` alone, so `base_lrs` is captured with **one** param group. The `log_vars` are added as a **later** group, and `_LRScheduler.step()` zips `param_groups` against `get_lr()`, whose length is `len(base_lrs)` = 1. `zip` truncates, so group 1's learning rate is never updated: it keeps the optimiser default 0.001 while the network decays. Measured in views-hydranet after 900 steps — network **6.011e-04**, balancer **1.000e-03**. Total balancer travel is then ~0.90 nats against the 2.6–4.1 nats its own stationary point requires, so unfreezing cannot reach that point on this vehicle at all. There is no error and no warning; the run completes and the weights are wrong. **Why this is a views-models concern and not only views-hydranet's.** All 10 HydraNet configs here declare `freeze_multitask_balancer: True`, so nothing in production is affected — but the **library default is `False`**, which means a new config that simply omits the key gets the broken path. Omission is not a neutral silence, and nothing read the key's absence. **Mitigated, not resolved:** the guard added in this entry's `Location` requires the key to be PRESENT and `True` for every HydraNet, so the next config cannot inherit the default by writing nothing. The scheduler defect itself is views-hydranet's and is deliberately **not** in its 0.1.3 release — changing training dynamics inside a correctness release is the C-112 mistake — so the resolution waits on a change there. **Separately struck, and not reopened by this entry:** unfreezing the balancer was ruled against unanimously by a 7-seat method review, and a pre-registered 3-seed bisect (views-hydranet `reports/results_balancer_bisect.md`) found unfreezing to be the cause of the C-113 runaway. If it is ever revisited, the scheduler must be fixed first. Cross-refs: **C-113** (the runaway unfreezing causes), **C-112** (behaviour changes riding on a correctness release), **C-159**/**C-160**/**C-161** (a check or a default that cannot see its subject reports a clean answer). |

---

### C-163 — C-120's "generalisable rule" recurred three times in one change, in a file that already cites it: a rule written in a register is not a rule the code enforces

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Adding a second declaration kind to a conformance test module — a new roster category, exemption set, or pair mapping — and widening existing guards onto it. Also: any future `@pytest.mark.parametrize` whose argument is a module-level collection that a later edit could narrow. |
| **Source** | `/code-review low` + `/review-diff` on PR #582 (views-models#566), 2026-10-18. Three of five independent reviewers converged on the same root. |
| **Status** | Mitigated |
| **Location** | `tests/test_roster_conformance.py` — the five pair guards widened to `PAIRED_MODELS` (`:932`, `:978`, `:1001`, `:1013`, `:1099`), `REPLICATE_FREE_KEYS` (`:171`), and `test_the_replicate_is_its_declared_seed` (`:1151`). The precedent that was not followed: `tests/test_roster_configs_load.py::_parametrized_scope` (`:114-134`). |
| **Notes** | **C-120 closed with a rule stated as generalisable — *"an escape hatch must be guarded at the same scope it operates"* — and PR #582 broke it three ways at once, in the sibling of the file that records it.** (1) **The real scope of a guard is its decorator, and nothing read the decorator.** Five pair guards were widened from `sorted(EXPERIMENT_ARMS)` to `sorted(PAIRED_MODELS)`; reverting any one of them is a one-line edit that leaves the full suite green. This is not hypothetical — `e874be0f` records the identical mutation in the sibling file as **SURVIVED** (*"TRACKED_MODELS narrowed to the roster -> SURVIVED"*), which is why `_parametrized_scope` exists there to read `pytestmark` rather than the variable. The new guard asserted on the **dict**, which is the mutation's blind spot. (2) **A third escape hatch, unpinned.** `ARM_FREE_KEYS` and `INSTRUMENTATION_KEYS` are both pinned as exact literals in both directions inside one consolidated test, added by #554 after review found *"the hatch was the entire attack surface"*. `REPLICATE_FREE_KEYS` was introduced with a **prose argument** that a pin was unnecessary — which is precisely the argument those pins replaced. (3) **A one-sided guard on a two-sided invariant.** `test_the_replicate_is_its_declared_seed` never required the declared seed to **differ** from its control's, and the seed-only diff guard permits differing on nothing, so a replicate carrying its control's own seed passed every check while contributing **zero** information to `σ̂` — the primary endpoint of epic #565 and the denominator of every threshold in the views-hydranet pre-registration. Silent, and indistinguishable from an honest measurement. **Why it is registered although it was fixed before merge:** the finding is not the three defects, it is that **C-120's rule had already been written down, cited in this very file, and still did not prevent its own recurrence.** A rule recorded in a register is a rule a future author must remember; the three guards added in the fix are rules the suite enforces. The transferable form: **when a declaration gains a sibling, audit the DECORATORS, the HATCHES and the INVERSE assertions — those are the three places scope silently fails to widen.** Fixed in PR #582 at ritual step 5 and proved by mutation after committing. Cross-refs: **C-164** (the SAME change's other half: one of the guards added by this fix was over-constrained, and the mutation proof could not see it), **C-120** (the prior, and the rule that recurred), **C-146** (a conformance test green against exactly the failures that matter — the sibling file's origin), **C-159** (a mutation that fails to apply reads as a result), **C-106** (STRATEGIC ROOT: declarations verified exhaustively, behaviour nowhere). Member of **Cluster A** (declared-but-unenforced). |

---


### C-164 — A mutation proof shows a guard CAN fire; it cannot show the guard should not fire on legitimate input — and the synthetic subject used to prove it satisfied the over-constraint

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Adding a guard whose real subjects do not exist yet, and proving it by mutation against a stand-in. Also: any assertion of the form "X must differ from its control" written before a real X has been read. |
| **Source** | `/code-review` on PR #583 (views-models#567), 2026-10-19 — found while building the first real subjects the #566 guards had ever seen |
| **Status** | Mitigated |
| **Location** | `tests/test_roster_conformance.py::test_the_replicate_is_its_declared_seed` (the withdrawal and its reasoning); `test_the_replicates_are_distinct_runs` (the invariant that replaced it) |
| **Notes** | **PR #582 added an assertion that a seed replicate's seed must DIFFER from its control's**, with the reasoning that a replicate sharing its control's seed is a byte-identical rerun contributing nothing to `σ̂`. ⛔ **The reasoning was sound and the assertion was wrong, because it assumed the control is a MEMBER of the variance cell.** views-hydranet's pre-registration §7 rules that out explicitly — `amber_kestrel` *"was trained on a different pod session at a different time under a different code sha"*, so pooling it would import every difference between then and now into the variance estimate — and §3's approved table assigns seeds **42 / 43 / 44** with `silver_kestrel` at **42**, which is amber's own. **So the guard would have refused the first real replicate it ever saw**, and the directory it refused was the one the pre-registration names. **Why the mutation proof did not catch it, which is the transferable part.** #582's guards were proved by mutation against a *synthetic* pair — `bright_starship` declared a replicate of `heavy_freighter` — because no real replicate existed yet. Those two carry seeds **43 and 47**. ⛔ **The stand-in happened to SATISFY the over-constraint, so every mutation came back "caught" and the proof was clean.** A mutation test applies a wrong input and checks the guard objects; it is silent about whether the guard also objects to *right* input, because supplying right input is not a mutation. **The two failure modes of a guard are not symmetric in what proves them: a mutation proves it is not DECORATIVE, and only a real subject proves it is not OVER-CONSTRAINED.** ⚠️ **And the cost is asymmetric too:** a decorative guard lets a defect through, while an over-constrained one blocks correct work and is removed under deadline pressure — after which it protects nothing and nobody remembers it was ever there. **The replacement invariant is pairwise distinctness ACROSS THE CELL, not relative to the control**, which is both correct for `silver` and strictly stronger: the control-relative form would have passed `bronze = cobalt = 43` as long as neither equalled amber's 42. ⛔ Independently reached by the views-hydranet seat as **Amendment 11**, which added the same requirement to **F13** as a third clause — its `(b)` *exempts* the seed keys from the identity check, and an exemption cannot express a requirement that they differ. ✅ **The direction argument is why this one matters:** every threshold is `C = k·σ̂`, so config divergence INFLATES `σ̂` (thresholds rise, effects fail — conservative, mimics the expected null) while a seed collision DEFLATES it (thresholds → 0, anything clears — **anti-conservative, manufactures a positive**). It is the first corruption on record in this plan that fails unsafe, and the argument needs no determinism assumption, which matters because Amendment 6 withdrew determinism as a precondition. **Mitigated:** assertion withdrawn with the reasoning kept in place, pairwise guard strengthened, and all five replicate guards re-proved by mutation **against the three real configs**. Cross-refs: **C-163** (the same change's other half — a rule stated but not enforced), **C-120** (escape hatch guarded at the wrong scope), **C-159** (a mutation that fails to apply reads as a result — the adjacent failure of the same proof method), **C-113**/**C-112**. Member of **Cluster G** (a check out of scope returns a clean verdict). **Moved from Cluster A during review-rr (2026-10-25)** — the defect here is a check whose verdict is uninformative, not a declaration nothing enforces. |

---


### C-165 — A declaration read only through `set(...)` has its VALUES guarded by nothing, and the key-set reads look like coverage

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Adding a mapping whose consumers only ever need its keys — `set(X)`, `for k in X`, `X[name]` — while its values carry the meaning. Also: declaring a new experiment arm, whose control is the comparison every result rests on. |
| **Source** | mutation testing on PR #584 (views-models#568), 2026-10-19. ⛔ **The mutation SURVIVED the first run; this entry is the fix.** |
| **Location** | `tests/test_roster_conformance.py::test_the_arm_roster_is_exactly_as_declared` (the pin); the unpinned reads were `set(EXPERIMENT_ARMS)` at `:982`, `:1248`, `:1310`, `:1331` |
| **Status** | Mitigated |
| **Notes** | **`EXPERIMENT_ARMS` maps an arm to `(control, declared_values)`. Four tests read it, and every one read it as a KEY SET or indexed it by name — so the CONTROL each arm is declared against was pinned by nothing in the repository.** `SEED_REPLICATES` and `ARM_FREE_KEYS` sit beside it and are both pinned as exact literals in both directions; this one was not, and nobody noticed because `set(EXPERIMENT_ARMS) == set(ARM_FREE_KEYS)` and `set(PAIRED_MODELS) == set(EXPERIMENT_ARMS) | set(SEED_REPLICATES)` *look* like the declaration is covered. **They cover its names. The meaning is in its values.** ⛔ **Demonstrated, not theorised:** re-pointing `indigo_kestrel`'s control from `silver_kestrel` to `amber_kestrel` — the exact confounded declaration the file's own comment at `:66-70` exists to forbid (*"NOT a one-variable pair: the seeds differ too"*) — left the **full suite green at 291 passed**. It survives because `indigo` and `amber` happen to share seed **42**, so the seed-pairing guard is satisfied, and the one-variable guard sees only `{weight_decay, trajectory_log_path}`, both admissible. **The declaration was wrong and every downstream check was right.** ⛔ **And green is the wrong answer twice over:** `amber_kestrel` was trained on a different pod session at a different time under a different code sha (prereg §7), so pairing a fresh arm against it reintroduces exactly the environment gap the three fresh replicates exist to exclude — the reason the pre-registration pairs each arm with a seed-matched *replicate*. **The hole was never specific to `indigo`:** `copper_kestrel`'s control re-points from `amber_kestrel` to `silver_kestrel` and also stays green, because `copper` vs `silver` differs on `{loss_class_pos_weight, trajectory_log_path}`, both admissible. Both re-pointings are now caught. ⚠️ **Why it was found only now, which is the C-164 lesson recurring:** `EXPERIMENT_ARMS` had held two arms for days, and the mutation that exposes this needs an arm whose control is *interchangeable with another declared model at the same seed* — a configuration that did not exist until the three seed replicates landed in #567. **A guard gap can be latent not because nobody looked, but because the input that reveals it did not exist yet.** **The generalisable rule: a declaration must be pinned at the granularity its MEANING lives at, not at the granularity its consumers happen to read.** Cross-refs: **C-166** (the other half of the same audit: the same declaration's free-key VALUES were unpinned, for a different reason), **C-164** (a mutation proof's blind spots — and the same "the revealing input did not exist yet" mechanism), **C-163** (a rule stated but not enforced), **C-120** (an escape hatch guarded at the wrong scope). Member of **Cluster A** (declared-but-unenforced). **Cluster G** (moved from Cluster A, 2026-10-26, #610) — the key-set read *looks like coverage* while the values are guarded by nothing. A check whose input is the key set cannot answer a question about the values, and it answers anyway. |

---


### C-166 — A declaration schema whose shape was fixed by its FIRST subject pins only that subject's variable: three of four experiment arms could silently run a different intervention

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Declaring an experiment arm whose variable is not one of the properties the declaration's tuple already carries. More generally: adding a second instance to a declaration whose field list was written for the first. |
| **Source** | mutation testing on PR #584 (views-models#568), 2026-10-19. ⛔ **Three mutations SURVIVED; this entry is the fix.** |
| **Location** | `tests/test_roster_conformance.py` — `ARM_FREE_VALUES`, `test_the_arm_is_its_declared_free_value`, `test_every_arm_declares_a_value_for_every_free_key`; the gap was between `ARM_FREE_KEYS` and `EXPERIMENT_ARMS`' declared tuple |
| **Status** | Mitigated |
| **Notes** | **`ARM_FREE_KEYS` licenses a key to differ from the control. It said nothing about what the key BECOMES — and for three of the four arms, nothing else did either.** ⛔ **Measured, not argued:** `indigo_kestrel`'s `weight_decay` set to **0.05**, then to **1e-12**, and `copper_kestrel`'s `loss_class_pos_weight` set to **1.7**, each left the whole file green at **292 passed** — while the config header, the README and the pre-registration all still said *"weight_decay 0.1 → 0.0"*. Setting `amber_kestrel`'s `output_distribution` to `'nb'` failed **5** tests. **THE ROOT, and it explains the asymmetry exactly:** `EXPERIMENT_ARMS`' declared tuple pins four properties — `(output_distribution, forecast_composition, gate_threshold, seed)` — and **that shape was chosen for the FIRST arm, whose variable WAS the composition triple.** Every arm declared since varies something outside the tuple, so its variable's value was pinned by nothing. **A free key's value was pinned if and only if it happened to be one of those four properties**, which is a property of the schema's history and not of anything anyone decided. ⛔ **Why this is Tier 2 and not hygiene:** the arms run on rented hardware and their output is read as the answer to a pre-registered question. A pod could have trained `weight_decay = 0.05` while every document in two repositories said `0.0`, and the readout would have reported it as the zero-decay result. **An arm running a different value is not a failed experiment — it is a successful experiment on a question nobody asked, reported as the answer to one they did**, and nothing downstream can detect it because the weights are the only record and they do not carry the config. **Fixed by `ARM_FREE_VALUES`**, which pins what each free key must be for every arm uniformly, with a sentinel for `amber_kestrel`'s `gate_threshold` whose correct state is **absence** rather than a value (ADR-069 reads `elif self.gate_threshold is not None`, so a `None` loads fine and invites someone to fill it in). ⚠️ **Two guards, because the pin alone would not stay true:** one asserts each arm's config against it *and* that the key is PRESENT — a free key the config never sets means the arm inherits the control's value and varies **nothing at all** — and one requires `ARM_FREE_VALUES` to cover `ARM_FREE_KEYS` **exactly**, so the next arm cannot declare a free key and forget its value, which is the state this mechanism exists to leave. That second guard also cross-checks the overlap with the declared tuple, since `amber`'s three free keys are now written down twice: two copies that cannot drift are fine, two that can are not. ✅ All six mutations caught after the fix, applied **after committing**. Cross-refs: **C-165** (the other half of the same audit — the same declaration's *control* was unpinned because it was only ever read as a key set), **C-164** (what a mutation proof cannot show), **C-163**, **C-120**. Member of **Cluster A** (declared-but-unenforced). **Cluster G** (moved from Cluster A, 2026-10-26, #610) — the schema's shape was fixed by its first subject, so its input excludes every later subject's variable. Identical in kind to C-164, which moved out of A on 2026-10-25 for the same reason. |


---

### C-167 — The replicate-identity check reads only the CONFIG, so a variance cell spanning a library-version boundary passes it and σ̂ reports the version difference as run-to-run noise

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Assembling a variance cell (seed replicates, or any set whose spread becomes a denominator) from runs that were not trained against a pinned dependency version — in practice, launching replicates on different days while a dependency is under active release. |
| **Source** | #572 M1-M4 readout, 2026-10-21. Measured on six arms after the pods were terminated; **the confound was then measured directly by a seventh arm (pod five) the same day and found inert — see Notes.** |
| **Location** | `tests/test_roster_conformance.py` — `SEED_REPLICATES` and the five `PAIRED_MODELS` guards; views-hydranet's **F13** (prereg `:1096`) and its Amendment 11 third clause. The dimension is in none of them. **Added 2026-10-24:** `models/*/requirements.txt` (the `views-hydranet~=0.1.1` specifier, all 15 HydraNet sources) and the generated-env freshness check in each `run.sh` — the two places that decide which version is actually installed. |
| **Status** | Open |
| **Notes** | **F13 asserts that replicates are byte-identical across `ARM_IDENTICAL_FILES` and identical in `CORE_GENOME` except the two seed keys. The library version is in neither, so a replicate set spanning a version boundary satisfies F13 completely.** ⛔ **Realised, not hypothetical:** epic #565's σ̂ cell is `{silver 42, bronze 43, cobalt 44}`. `silver` trained on **views-hydranet 0.1.3**; `bronze` and `cobalt` on **0.1.4**. Every conformance test passed, F13's assertion holds as written, and `silver`'s seed is correctly distinct — the set is defective in a dimension nothing in either repository records. **The contamination is measurable because this pipeline turned out to be deterministic.** `amber_kestrel` (0.1.2) and `silver_kestrel` (0.1.3) are the same config at the same seed and produced **190/190 bit-identical weight tensors** and a gate cube equal under `np.array_equal`, so stochastic run-to-run variance is **zero** and pod hardware is excluded. With no noise term, the spread in the σ̂ cell is seed + version and nothing else — and `bronze` vs `cobalt` isolates seed at fixed version at **σ̂_b = 0.0002–0.0196**, against **0.0351–0.0932** for the mixed cell. This entry originally concluded from that gap that **"4–200× of the primary endpoint is attributable to a dependency version by elimination."** ⛔ **MEASURED AND REFUTED, 2026-10-21.** `silver_kestrel` was retrained at seed 42 on **0.1.4** — same config, same git sha `2e73de3`, nothing else moved — and produced **190/190 bit-identical weight tensors** and a gate cube equal under `np.array_equal` to the 0.1.3 run. **`0.1.3 → 0.1.4` is numerically inert for this config**, so the clean cell `{42, 43, 44}` all on 0.1.4 returns *exactly* the as-launched σ̂ (0.0374 / 0.1107 / 0.0463 on three-arm common support). The elimination argument was structurally valid and empirically wrong: `bronze` vs `cobalt` measures the 43↔44 distance, which is small, and says nothing about how far **42** sits from either. The excess was the seed all along. ⚠️ **The direction argument C-165 established is incomplete, and that is the transferable part.** C-165 holds that config divergence INFLATES σ̂ and is therefore conservative, because every threshold is `C = k·σ̂` and inflation makes effects fail to clear. **That reasoning is sound only for the hypothesis tests.** Epic #565's DoD item 5 is a claim that σ̂ is *small* — a feasibility statement — and for that claim inflation is **anti-conservative**: it reports a design as unable to detect a complete fix when the uncontaminated estimate clears the boundary comfortably. ⛔ **The claimed harm did not materialise, and the correction ran the wrong way.** This entry originally held that the readout's *"under every reading, this design cannot detect a complete fix"* was anti-conservative, and that on the same-version pair the design "probably can" detect a complete fix. The clean three-seed cell says otherwise: σ̂_b = 0.0374 / 0.1107 / 0.0463, **all three above the 0.0356 feasibility bound**. The original robust negative was correct; this entry's correction of it was the error. The 1-df `{43,44}` pair that suggested otherwise was two adjacent seeds, not a noise floor. **Inflation is conservative for a difference and anti-conservative for a bound; which one applies depends on the endpoint, not on the corruption.** ⚠️ **Why no guard could have caught it in this repository:** the version is not a config field, not in `CORE_GENOME`, and not written into `PROVENANCE`. It is recoverable only from `views-hydranet==0.1.x` in the pod's `run.log` — an artifact nothing parses. A check is possible (assert one version across a variance cell, read from the run log or recorded at train time into the sidecar) but it has to be built; there is nothing to widen. ⛔ **And the root is narrower than "pin your dependencies":** the replicates were launched on three different days *while 0.1.3 and 0.1.4 were being released*, and the launcher installs the latest. **A variance cell assembled over time against a moving dependency is confounded by construction, whatever the configs say.** ⛔ **What survives, and why this stays Open at Tier 2.** The hole is untouched: the library version is still absent from the sidecar, from `CORE_GENOME`, from `ARM_IDENTICAL_FILES` and from `PROVENANCE`, so a variance cell spanning any future boundary still satisfies F13 completely and silently. What pod five establishes is that **this** boundary carried no signal — 0.1.2, 0.1.3 and 0.1.4 produce one model for this config. It does not establish that the next one will. The lesson is the inverse of the one first recorded here: **an unrecorded dimension is a reason to measure, not a licence to attribute.** The attribution was published to #572 before the free check that settled it had been run, and the check cost $8.50 plus a pod. ⛔ **2026-10-24, `/falsify`: the mechanism is now measured at the pin, and it does not need a variance cell to bite.** All 15 HydraNet sources declare `views-hydranet~=0.1.1`, which means `>=0.1.1, ==0.1.*` — it admits **every** 0.1.x, including **0.1.7, published 2026-10-23T23:53Z**. `envs/views-hydranet` holds **0.1.5**. Each `run.sh` decides whether to upgrade with `pip install --dry-run -r requirements.txt`, counting lines that are not *"Requirement already satisfied"* — and 0.1.5 satisfies the range, so **an existing env is left alone while a freshly created env resolves 0.1.7.** Which library code runs is therefore decided by the age of the environment, not by anything the repository declares, and rented hardware always gets the newest. That makes two runs of a byte-identical config non-comparable **without any replicate set being involved**, which widens this entry from variance cells to every HydraNet run. The configs survive the jump today — diffed `HydraNetConfig` between 0.1.5 and 0.1.7: **0 fields removed, 2 added** (`emit_distribution`, `trajectory_log_path`, the latter being the key #591 was opened about) — so nothing is broken right now. **The finding is that nothing in the repository would have reported it if the diff had not been additive.** Guard at `tests/test_falsification_heavy_freighter_launch_readiness.py` (red by design; it asserts the pin names a version it was validated against, and carries `EXPECTED_HYDRANET_SPEC_IS_EXACT = False` as the explicit switch if the policy is rejected). ✅ **2026-10-25 — the `run.log` recovery path was exercised for real, and it works but does not scale.** The `heavy_freighter` control run was launched on a fresh pod, so it resolved **views-hydranet 0.1.7** as this entry predicted a fresh env would. Confirming which build trained it required grepping the pod transcript — `run.log:301`, `+ views-hydranet==0.1.7`, exactly one install line in 5.6 MB — and nothing else in the delivered bundle records it. views-hydranet's session asked for that grep as a **time-critical, pod-lifetime-only** action (its S9, `reports/2026-10-24_control_vehicle_prereg.md`); it was in fact already recoverable because `run.log` ships inside `$OUT`, which is the one piece of luck in this mechanism. ⚠️ **Partial mitigation, for one run only:** the full 172-package pinned set was extracted from `run.log` and written to `~/views-diagnostics/hf_control_artifact/ENVIRONMENT`, carrying `run.log`'s own sha256 plus the git sha and the train window so the attribution is checkable rather than asserted (views-hydranet 0.1.7, views-pipeline-core 3.3.4, torch 2.14.1, numpy 1.26.4, pandas 1.5.3). **That is a hand-made file for a single run, not a runner behaviour — `PROVENANCE` still records no package versions, so the next run is in exactly the same position.** The cheap fix this instance argues for: have the manifest stage parse its own install output, which it already has in front of it, rather than leaving every future reader to re-derive it from a transcript that only survives by accident. Cross-refs: **C-165** (the other F13 hole — seed collision, the within-config case, and the source of the direction argument this entry corrects), **C-163** (a rule stated but not enforced), **C-166** (a declaration pinned at the wrong granularity), **C-106** (STRATEGIC ROOT: declarations verified exhaustively, behaviour nowhere). Member of **Cluster A** (declared-but-unenforced). **Cluster G** (moved from Cluster A, 2026-10-26, #610) — the replicate-identity check reads only the config, so its input excludes the library version and a cell spanning a version boundary passes. The pass is then used as the denominator. |


---

### C-168 — CLUSTER ROOT: completeness is verified against the producer's own declared scope, so a flawless verification pass certifies an incomplete deliverable

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Declaring a pod, container or any rented host ready to destroy on the strength of a bundle-vs-manifest comparison — or reporting "we have everything" after a file-by-file diff against the collection script's own output directory. |
| **Source** | `/falsify` on "we are ready to terminate the pod", 2026-10-25, during the `heavy_freighter` 300-lesson control run |
| **Status** | Open |
| **Location** | The retrieval procedure, not a file. Nearest code: `tools/podrun/pod_run_model.sh` (`OUT=$ROOT/deliver/$MODEL` at `:97`, and every stage that writes into it); `docs/runpod_run_guide.md` Phase 5 (the rsync of `$OUT`, and the "~22 MB / abort if gigabytes" sizing that encodes the same assumption) |
| **Notes** | **Every completeness check we run compares the laptop against `$OUT`, and `$OUT` is the runner's OWN definition of the deliverable. A deliverable cannot audit its own scope.** ⛔ **Measured, with the pod live and the operator poised to destroy it.** Three probes passed perfectly and proved nothing: an inventory diff of `$OUT` (**25/25** identical by name and size), an inventory diff of the plot tree (**2729/2729**, both diagnostics timestamp dirs), and sha256 of every file on both sides (**2754/2754 byte-identical**). On that evidence this seat told Simon *"nothing left on the pod that you need — you can kill it."* The one probe that could have caught anything was the omission sweep, and it found **five artefact classes that existed only on the pod**: the 6 computed metricframes (the scores), 18 W&B evaluation-metric tables, the `by_*` gate posterior (6.55 GB raw), the training input with its only provenance pin (`queryset_digest`), and the pipeline's own `DEBUG`/`INFO`/`WARNING` logs. Three of the five were rescued minutes before termination. ⛔ **The pattern is the point, and Simon named it before the audit did:** *"every time you say I can terminate the pod, I ask you to falsify. And every time it takes one or two passes before everything is done."* Three separate "ready to terminate" statements in one session, each followed by a falsification pass that found more. **The claims were not careless — they were backed by the most thorough verification available, which is exactly why they were believed.** ⛔ **And the register already held the answer.** **C-161** was opened 2026-10-13 with the trigger *"Destroying a rented pod after `pod_run_model.sh` reports `STATUS: OK`, for any model whose readout depends on the gate"* — a verbatim description of what happened twelve days later. It was not consulted before the pod was declared safe. **A register entry that predicts an incident and is not read is indistinguishable from an entry that does not exist**, which is C-154's lesson (a citation resolving to nothing stops the next reader looking) arriving from the other direction. ✅ **The mechanism of the fix, which is what makes this entry actionable rather than a lament: a run's output is *every file it touched*, established by an mtime sweep rooted at the run's start time — not *the directory the collection script copies*.** Concretely, before any terminate: `find <each mount> -newermt "<train start>" -type f`, classify every hit, and account for each one as collected / known-duplicate / deliberately-discarded. On the control run that sweep reduced to 2,861 files in six classes and left exactly one unexplained hit (`.git/index`, touched by the audit's own `git status`). **The sweep must be rooted on each mount separately — see C-169, which is how it silently returned nothing the first time.** ⚠️ **Scope note:** `tools/podrun/pod_run_model.sh` is read-only by standing instruction, so this is a runbook and procedure fix, not a code fix. The runner is not wrong; it collects what it says it collects. Cross-refs: **C-161** (the sharpest instance — and the entry that predicted this one), **C-169** (the sweep's own blind spot), **C-170** (the four classes outside `$OUT`), **C-171** (a stale file inside `$OUT` read as current-run progress), **C-174** (output lands outside the repo, so the repo cannot be swept for it), **C-106** (STRATEGIC ROOT: declarations verified exhaustively, behaviour nowhere — the same shape one layer down, in CI rather than in retrieval), **C-159**/**C-160** (a check that cannot read its input reports a clean answer), **views-hydranet C-317** (a lost posterior needs a new GPU run). **Root of Cluster G** (a check out of scope returns a clean verdict) — see [Causal clusters](#causal-clusters). **Retagged during review-rr (2026-10-25):** this entry was registered as a member of Cluster A while its own title declared it a cluster root, and the ten entries registered beside it as its symptoms carried no cluster tag at all — so the root rooted nothing, and C-161, the entry whose trigger had described the incident twelve days before it happened, was in no cluster either. That is this entry's own mechanism applied to its own registration: the registration was verified against the register-risk skill's declared scope (entry schema, header counts, dangling references) and passed, while the property that makes an entry reachable before a destructive action went unbuilt. |

---

### C-169 — `find -xdev` silently excludes `/workspace` on a RunPod host, so the omission sweep that exists to catch what was left behind returns nothing and reads as a clean bill of health

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Rooting an mtime sweep, a disk-usage audit or any "what else is here?" search at `/` on rented hardware — or reusing a one-liner that carries `-xdev`, `-mount` or `--one-file-system` from a laptop context to a pod. |
| **Source** | `/falsify` on "we are ready to terminate the pod", 2026-10-25 — the audit's own instrument failing mid-audit |
| **Status** | Open |
| **Location** | The sweep command, not a file. The property that defeats it: `/workspace` on RunPod is a network mount (`mfs#us-ks-2.runpod.net:9442`, 873 T), a different device from `/`. |
| **Notes** | **`-xdev` tells `find` not to cross device boundaries. On a RunPod host everything that matters — the repo, the model directory, `$OUT`, 13 GB of draws — is on `/workspace`, which is a separate device. So a sweep rooted at `/` with `-xdev` searches everything *except* the data.** ⛔ **Measured, during the audit it was meant to serve.** The omission probe was `find / -xdev -newermt '2026-10-25 10:43:00' -type f` with the known-collected paths filtered out. It returned **empty**, and empty is precisely the output that says "nothing was left behind". The whole-filesystem count was **43 files** at a moment when a single directory on `/workspace` held **156 draw files totalling 13 GB**. ⚠️ **What caught it was not the instrument.** It was knowing that 13 GB of draws had to be *somewhere* and not seeing them in the output — i.e. an expectation held outside the tool. Had the run's output been something whose size was not already known, the empty result would have stood. **This is C-159/C-160's rule — "a check that cannot read its input must FAIL, not pass" — firing on the audit apparatus rather than on production code**, and it is the second time in this register that the instrument rather than the subject was the defect. ✅ **The fix is to root the sweep per mount and to refuse an empty result as an answer:** enumerate mounts (`findmnt -rn -o TARGET`, or at minimum `/` and `/workspace`), sweep each separately without `-xdev`, and treat a zero-hit sweep as **UNRUN** until a positive control confirms the sweep can see a file known to be there. On the control run the corrected sweep returned 2,861 files in six classes; the broken one returned nothing, and the two commands differ by one flag. ⚠️ **Generalisable beyond `find`:** any tool whose search domain is set by a flag rather than by the question — `du -x`, `rsync --one-file-system`, `grep -r` from the wrong root — fails the same way, silently and in the safe-looking direction. Cross-refs: **C-168** (ROOT — this is the blind spot in its prescribed fix), **C-159** (a mutation that silently fails to apply reads as a result — the same "instrument's own input unverified" root), **C-160** (a check that cannot read its input reports a clean answer), **C-154** (the other `/workspace`-is-not-a-normal-filesystem defect: `chmod` there returns success and does nothing). Member of **Cluster G** (a check out of scope returns a clean verdict). ⛔ **2026-10-26 — SECOND INSTANCE, by a different mechanism, inside the tool written to remedy the first.** `tools/podrun/pod_sweep_outputs.sh` implements this entry's prescribed fix — its header states *"Sweeps EACH MOUNT SEPARATELY, so `-xdev` is correct instead of silently excluding"* — and then excludes the container's entire root filesystem by a second route: `overlay` appears in its `PSEUDO` regex of filesystems not worth sweeping, and on a RunPod container **`/` IS an `overlay` mount**. Run against the #607 arm pod it reported one verified mount (`/workspace`) and never examined `/` at all. A manual `find / -xdev -newermt` found **38,510 files** from that run there (6.27 GB — pip/uv/HF cache, wandb artifact staging, apt and dpkg logs; **no run output**, verified by excluding known cache paths and reading what remained). ⚠️ **The loss was zero and the defect is not:** the blind spot was real, it was invisible to the sweep's own verdict, and it was closed only by a human-written one-liner run outside the tool — the same thing that saved the first instance. **One exclusion was a flag, the next was a filesystem-type list; the shape is that the search domain is set somewhere other than the question.** ✅ The fix: drop `overlay` from `PSEUDO` (a container root holds real files), and more generally allow only *pseudo* filesystems that cannot hold persistent files to be excluded by type — `overlay`, `squashfs` and `fuseblk` all can. See **C-181** for the companion defect found in the same tool on the same run. |

---

### C-170 — Four classes of run output are written outside `$OUT`, so the collection step cannot see the scores, the W&B tables, the training input or the pipeline logs

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Destroying a pod after collecting `$OUT`, when the readout needs a computed metric, a W&B table, the input the model was trained on, or a pipeline log — i.e. any run whose purpose is to be *scored* rather than to produce weights. |
| **Source** | `/falsify` on "we are ready to terminate the pod", 2026-10-25 (`heavy_freighter` control run) |
| **Status** | Open |
| **Location** | Written by the library into the model directory, outside the runner's collection scope: `models/<model>/data/generated/<model>/calibration/metricframe_{lr,by}_{sb,ns,os}_best/`; `models/<model>/wandb/offline-run-*/files/media/table/`; `models/<model>/data/raw/`; `models/<model>/logs/`. Collection scope is `tools/podrun/pod_run_model.sh:97` (`OUT=$ROOT/deliver/$MODEL`) and the Phase 5 rsync in `docs/runpod_run_guide.md`. |
| **Notes** | **`$OUT` holds the weights, the parquets, the `lr_*` draws, the trajectory and `run.log`. It holds none of the following, and all four were on the pod and nowhere else when this seat reported the bundle complete and verified.** ⛔ **(a) The computed scores.** `data/generated/<model>/calibration/metricframe_{lr,by}_{sb,ns,os}_best/` — six metricframes (`values.npy`, `identifiers.npz`, `metadata.json`), written **14:12:00–14:14:02** with `run_id szwc8tzk` matching the control run's W&B offline run, and `metadata.json` carrying `data_version 504` / `scoring_code_version 2.0.0`. **These are the evaluation metrics.** C-161 already records that `metricframe_by_*` holds the gate's only metric (`Brier_cls_sample`) and `metricframe_lr_*` holds `CRPS`/`MCR_sample`/`QS_sample` — so the entire scored result of a calibration run lives in a directory the collector never reads. ⛔ **(b) 18 W&B evaluation-metric tables.** `wandb/offline-run-20261025_132148-*/` and `-20261025_132154-*/files/media/table/evaluation_metrics_{ts,step,month}_*.table.json`, i.e. the per-horizon metric tables, offline because the pod has no W&B credentials. Note the companion hazard already recorded at **C-84**: `wandb/` divergence after a partial re-run breaks the ensemble evaluation report. ⛔ **(c) The training input and its only provenance pin.** `data/raw/calibration_datafactory_df.parquet` (64.7 MB) plus `.provenance.json` — `queryset_digest a7fce3a6cfdc4c8b5dbb46be37a3a1ddc75f3c80ae090d06cfc5c5928a1bc685`, `level pgm`, `partition calibration`, `month_first 121`, `month_last 504`, `source datafactory` — and `calibration_data_fetch_log.txt`. **That digest is the only record of what the model was actually trained on**, and `drift_detection_ran: false` is recorded in the same file. Without it a run is reproducible only in the sense that it can be re-fetched and hoped to match. ⛔ **(d) The pipeline's structured logs.** `logs/views_pipeline_{DEBUG,INFO,WARNING}.log` (3.9 MB / 1.5 MB / 5.6 KB). `run.log` is the runner's stdout and is *not* a superset — the `WARNING` log is where `aggregate_method is IGNORED` and the data-sniffer anchor warnings are legible rather than buried in tqdm output. ⚠️ **Why the collector is not simply wrong:** `$OUT` is a deliberate, documented staging directory and the runner fills it correctly. The gap is that the *library* writes its most interpretable output into the model tree by its own conventions, and no document reconciles the two lists. **The remedy is on the retrieval side (sweep, per C-168) or in the runbook, not in the read-only runner.** ✅ All four classes rescued 2026-10-25 and verified: **95/95** files by content hash (`~/views-diagnostics/hf_control_metrics/`), **11/11** (`~/views-diagnostics/hf_control_inputs/`). Cross-refs: **C-168** (ROOT), **C-161** (the fifth class — the gate posterior — and the entry that predicted the near-miss), **C-84** (`wandb/` divergence), **C-110** (config reproducibility), **C-167** (the package versions, also absent from the bundle and recoverable only from `run.log`). Member of **Cluster G** (a check out of scope returns a clean verdict). |

---

### C-171 — `$OUT` persists between runs on one pod and the startup clear misses `logs/` and `artifact/`, so a stale file reads as current-run progress and a guardian "secures" the previous run's weights

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Writing any monitor, guardian, progress reporter or readiness check that keys on the **existence** of a file under `/workspace/deliver/<model>/` — in particular after a rehearsal, a failed attempt or any earlier run on the same pod. |
| **Source** | Three self-inflicted instances in one session, 2026-10-25, during the `heavy_freighter` rehearsal → control-run sequence |
| **Status** | Open |
| **Location** | `tools/podrun/pod_run_model.sh:130-144` (the startup clear: `STATUS`, `REHEARSAL`, `.lessons`, `.region`, `FAILURE` — **not** `logs/` or `artifact/`); `:395-396` (`collect_artifact`, whose `rm -rf "$OUT/artifact"` is the only thing that clears it); the library's `open(path, "w")` on `trajectory_log_path` at the lesson loop |
| **Notes** | **`$OUT = /workspace/deliver/<model>` survives across runs on one pod. The runner clears five status files at startup but leaves `logs/` and `artifact/` standing until their own stage rewrites them — so between launch and `collect_artifact` (≈2h40m on a 300-lesson run) the directory contains the PREVIOUS run's trajectory and the PREVIOUS run's weights, under the exact filenames the current run will eventually use.** ⛔ **Three instances, same day, same root.** **(i)** Progress reported as *"5 of 300 lessons"* when the true count was **0** — the `trajectory.csv` being read had mtime 09:42:24, from the rehearsal, because `$OUT/logs/` was never cleared. **(ii)** An artifact guardian keyed on `test -f "$OUT/artifact/PROVENANCE"` fired immediately, set `SAVED=1`, pulled 4.1 MB and reported **"ARTEFACT SECURED LOCALLY"** — having secured a 5-lesson rehearsal artifact whose own `PROVENANCE` read `lessons: 5` and `mode: REHEARSAL -- NOT FIT FOR MEASUREMENT OR DELIVERY`. **The guarantee the guardian existed to provide — weights pulled the moment they are written, ~2 h before the run ends — was silently void for the whole run.** **(iii)** A freshness check hardcoded `[ "$MT" -gt 1761388998 ]` when the correct epoch for the train start was `1792924998` — **off by about a year** — so the stale file passed as FRESH and the watcher exited at once, reporting success. ⚠️ **Why existence is the wrong key even when the clear is correct:** a file that *will* be written is indistinguishable from one that *has* been written by any test on its path, and the two differ by hours. The question a monitor asks is never "does this path exist" but "does this path hold THIS run's output". ✅ **The fix that worked, and it needs both halves:** key on **content** *and* on **mtime against this run's train start**, with the epoch computed **on the pod** rather than on the laptop. For the artifact guardian that is `mtime(PROVENANCE) > train_start && grep -q '^mode: *production' PROVENANCE`; the corrected guardian then correctly reported `secured=0` for the whole training phase and fired only when real production weights appeared. **Computing the epoch on the pod is not a detail** — instance (iii) is what happens when it is computed anywhere else. ⚠️ **Same missing-run-identity root as C-14, one directory further on:** C-14 records that artifacts have no run ID and silently overwrite; this entry is the *consumer* side on rented hardware, where the overwrite has not happened yet and the old file is read as the new one. C-85 is the same consumer-side defect for the ensemble, and C-97 for delivery selection by recency. Cross-refs: **C-14** (no run identity on artifacts — the root), **C-85** (stale constituent predictions read as current), **C-97** (selection by recency, not identity), **C-168** (ROOT — presence of a file as proof of provenance is the same unverified-scope error), **C-161** (the run reported `OK` while half the posterior was still on the pod). Member of **Cluster H** (no identity separates current from left-over). |

---

### C-172 — `--rehearsal` cannot clear the collapse stage, so the runner's claim that it "exercises the whole chain" is false for the stage most likely to fail

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Running `pod_run_model.sh --rehearsal` to validate the pipeline before a paid run, and reading a `FAILED:collapse` as a defect in the configuration or the model rather than as the guard correctly refusing an undertrained output. |
| **Source** | Observed 2026-10-25 on the `heavy_freighter` 5-lesson rehearsal, which reached collapse and died there |
| **Status** | Open |
| **Location** | `tools/collapse/collapse_predictions.py:175-197` (`_check_scale`, `MIN_PLAUSIBLE_MAX = 12`, and the `--min-plausible-max` CLI flag); `tools/podrun/pod_run_model.sh` (the collapse stage invocation, which never passes that flag) and the header comment claiming the rehearsal exercises the whole chain |
| **Notes** | **`_check_scale` refuses any `pred_*` column whose maximum is below 12, on the grounds that such a column never left log1p space. A 5-lesson rehearsal is undertrained by construction, so its predictions are small, so the guard fires — correctly — and the rehearsal can never reach a green collapse.** Measured: the rehearsal died with *"largest collapsed 'pred_lr_sb_best' is 4.688, below 12"* after completing train (5 lessons), evaluate (13 origins, 41 min) and 13 parquets. ⚠️ **The guard is right and should not change.** Its own docstring says so: *"The trade-off is deliberate: a genuinely tiny target trips a false alarm and stops the conversion. That is the safe direction."* It is checked per target rather than on the three flattened together, which is also right. **The defect is the runner's claim, not the guard's behaviour.** ⛔ **And the two halves do not know about each other:** `grep -ic rehearsal` in `collapse_predictions.py` is **0** — the collapse tool has no concept of a rehearsal — while `pod_run_model.sh` has no concept of the scale floor and never passes the `--min-plausible-max` flag that exists precisely to relax it. So a flag that would make the rehearsal meaningful exists, is reachable, and is wired to nothing. ⚠️ **What it costs, which is why this is Tier 3 and not Tier 4:** a rehearsal is run to buy confidence before spending money on a pod, and this one ran **09:36:57 → 11:32** (~2 h, including a 41-minute eval) to end in a red stage that is red by design. An operator who reads `FAILED:collapse` as a real defect either debugs a non-problem or — worse — loses trust in a rehearsal that was otherwise genuinely informative. ✅ **This rehearsal earned its cost regardless**, which is the honest counterweight: it proved `matplotlib` present in the venv, measured plot volume (54 files / 12 MB per 5 lessons, extrapolating to ~572 MB at 300), measured real link throughput at **2.33 MB/s against the run guide's documented 43–48 KB/s** (~50×, which turned "curate a subset of plots" into "pull everything"), and produced plots the operator could inspect and approve before authorising the paid run. ⚠️ **Scope:** `pod_run_model.sh` is read-only by standing instruction, so the remedy is either passing `--min-plausible-max` from the runbook, or correcting the header's claim. Cross-refs: **C-161** (the other place the runner reports a status its scope cannot justify), **C-168** (ROOT), **C-144** (a coherence rule whose `check()` has no caller outside the test suite — the same "mechanism exists, nothing invokes it" shape). Member of **Cluster A** (declared-but-unenforced). **Cluster G** (moved from Cluster A, 2026-10-26, #610) — the rehearsal's scope excludes the stage most likely to fail, and the runner reports it as having exercised the whole chain. A narrowed scope reported as a complete one. |

---

### C-173 — "The config as run is recorded" is true of a delivered bundle but not visible in it: the sidecar carries 12 architecture keys, not the 63-key config

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A recipient — views-hydranet, a reviewer, or a future reader of a delivery note — being asked to confirm from a delivered bundle alone that the run used the intended configuration. |
| **Source** | `/falsify` against the `heavy_freighter` launch-readiness claim, 2026-10-24/25, checking epic DoD criterion 4 |
| **Status** | Open |
| **Location** | `<artifact>.pt.config.json` (the sidecar); written by views-hydranet `train/train_model.py:85` (`config_snapshot = {k: config[k] for k in arch_keys}`); `tools/podrun/pod_run_model.sh:445` (`PROVENANCE`) and `:244` (`check_config`) |
| **Notes** | **The claim is TRUE. It is just not checkable from the artefact a recipient receives, and the field that looks like it should settle it does not.** The `.pt.config.json` sidecar holds **12 keys** — `model, input_channels, total_hidden_channels, output_channels, dropout_rate, static_channels, static_top_skip, torch_seed, np_seed, output_distribution, reg_activation, n_head_samples` — not the 63 keys of `config_hyperparameters.py`. Verified on the control run's sidecar. Notably **`trajectory_log_path` is absent from it**, so the instrumentation the run was configured for cannot be confirmed from the sidecar either. ✅ **What actually makes the claim true, and it is sound:** `PROVENANCE` records `git: <short sha>` (`c8ae1803` for the control run), and `check_config` **refuses to proceed on a non-rehearsal run if `config_hyperparameters.py` is dirty** — so the committed config at that sha *is* the config as run, and the pod's working tree was confirmed clean at `c8ae1803`. The chain is two files and one refusal, and it holds. ⛔ **But nothing in the bundle says so.** A recipient holding `artifact/`, `parquet/`, `draws/`, `MANIFEST` and `PROVENANCE` sees a 12-key sidecar and a git sha, and has no way to know that the sha plus a guard they cannot see is the thing that licenses the claim. The natural reading of a file named `*.pt.config.json` is "the config", and it is a twelfth of it. ⚠️ **This is the recipient-effort rule, which this repository has already paid for once:** a deliverable must not ship a verification step the recipient has to run or reconstruct. Asking views-hydranet to reason about `check_config`'s dirty-tree refusal in order to trust a DoD item is exactly that. ✅ **Cheap fix available and partially exercised:** the control run's bundle now also carries an `ENVIRONMENT` file derived from `run.log` (see C-167), and the same stage could as easily emit the 63-key config it already loaded and *called* at `check_config` time — it has the dict in hand. **The information exists at the moment the guard runs; only the writing-down is missing.** Cross-refs: **C-167** (the same bundle-opacity defect for package versions, and the same remedy), **C-170** (four further classes the bundle does not carry), **C-110** (config reproducibility: the working tree differs from git), **C-156** (`spec_from_file_location` can read stale bytecode, so even "the config as loaded" has a subtlety), **C-168** (ROOT). Member of **Cluster A** (declared-but-unenforced). |

---

### C-174 — A pod run's output lands outside the repository tree, so searching the repo for it returns a stale answer with no error

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Any session or script trying to locate a pod run's output by searching the repository — or inferring "no run happened" / "the run was lost" from repository state. |
| **Source** | The views-hydranet session's `/falsify` F1, 2026-10-25, relayed cross-session and resolved the same day |
| **Status** | Open |
| **Location** | `docs/runpod_run_guide.md` Phase 5 (the rsync target is outside the repo by design); the convention `~/views-diagnostics/<model>_<purpose>_<date>/`; the repo-side decoys are `models/*/data/generated/` and `models/*/artifacts/` |
| **Notes** | **Retrieval deliberately writes outside the repository, so the repository is never evidence about what a pod produced — and a sweep of it answers confidently and wrongly.** ⛔ **Realised cross-session, same day.** The views-hydranet session swept `views_platform/` for the control artefact, found the newest `.pt` anywhere was **2026-10-21** (`models/silver_kestrel/data/generated/pod_delivery/`, four days old and the wrong vehicle), observed that nothing in views-models had changed since 2026-10-23 outside `reports/` and `scratchpad/`, and reported: *"Whatever it produced has not come down to this machine. I cannot tell from here whether it is still training, finished and undownloaded, or lost."* It then flagged a **time-critical, pod-lifetime-only** recovery action on that basis. **The complete, 2754-file, content-hash-verified bundle was on the same disk at `~/views-diagnostics/hf_control_artifact/` the whole time**, and the perishable item was already inside it. ⚠️ **The reasoning was sound and the premise was invisible.** "Newest `.pt` in the repo is four days old" licenses "no run landed" only if runs land in the repo, and that is exactly the convention Phase 5 breaks — for good reasons (`models/*/data/generated/` is gitignored churn, and C-112/C-105 record what shared-checkout drift costs). **The defect is that the convention is recorded in a runbook and nowhere a searcher would look.** ⛔ **Why it is more than one session's mistake:** the repo contains *plausible* wrong answers rather than nothing. `models/*/data/generated/pod_delivery/` and `models/*/artifacts/` exist, hold real `.pt` files from earlier runs, and are the first thing any sweep finds — so the search returns a confident, dated, wrong artefact instead of an error. That is C-171's "presence is not provenance" at repository scale. ✅ **Cheap fixes, none applied:** have the runbook's Phase 5 note the destination convention somewhere greppable from the repo root (a line in the model README, or a `reports/delivery_notes/` pointer per run); or treat `~/views-diagnostics/` as a declared location in `CLAUDE.md` so a session knows to look there before concluding absence. **A sibling session cannot read this session's transcript, so anything not written down is not knowledge the platform has.** Cross-refs: **C-168** (ROOT — the same unverified-search-domain error, here in the laptop's tree rather than the pod's), **C-169** (a sweep whose domain silently excludes the data), **C-171** (presence of an artefact is not evidence of which run made it), **C-105** (the shared checkout drifts durably behind `origin/development`), **C-97** (selection by recency, not identity). Member of **Cluster G** (a check out of scope returns a clean verdict). |

---

### C-175 — A default value, a warning message, or an absent key is read as proof that a config knob is dead, without grepping for its consumer

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Describing any config key as inert, dead, a no-op or "not wired" — or answering "is \\<feature\\> on?" from a schema default plus one config file. |
| **Source** | Three instances in one session, 2026-10-25, two of them wrong; found when a peer session contested the third |
| **Status** | Open |
| **Location** | The diagnostic procedure, not a file. The three subjects: `models/*/configs/config_meta.py` + `views_pipeline_core/.../configuration.py:460-492` (`diagnostic_visualizations`); `views_hydranet/utils/inference_orchestrator.py:179` (`aggregate_method`); the 8 remaining `h_init` declarations (**now 0** — deleted 2026-10-27, #630/PR #632) |
| **Notes** | **A declaration tells you a key exists. A default tells you what it is when unset. A warning tells you someone anticipated confusion. *None of the three tells you whether anything reads the key*, and that is the only question that matters.** ⛔ **Instance 1 — `diagnostic_visualizations`, WRONG, and it destroyed a run.** Read `config_initializer.py:436` (default `False`) and `config_hyperparameters.py` (key absent), concluded OFF, reported OFF. Never opened `config_meta.py`, which sets it `True` for all 15 HydraNet models, and never checked pipeline-core's merge order, which makes `config_meta` win **silently**. Simon killed a healthy, running control run on that statement. Full mechanism at **C-68**, which this session also found to be inverted. ⛔ **Instance 2 — `aggregate_method`, WRONG, and caught only by a peer.** The control run logged a boxed `HydraNetConfig WARNING: aggregate_method is IGNORED` four times, and this seat reported it to Simon as a third inert knob *"in `h_init`'s class"*. It is not: `inference_orchestrator.py:179` is `pred_handler = pred_handler.collapse_to_point(method=self.config["aggregate_method"])` — the key is **read**, is **live** under `evaluation_mode='point'`, and is inert only under `'stochastic'`, with a dedicated validator (`config_initializer.py` `warn_aggregate_method_in_stochastic_mode`) and a pinning test. **A documented mode interaction, not a dead knob**, and the loud warning was evidence of care being taken, which was read as evidence of breakage. ✅ **Instance 3 — `h_init`, CORRECT.** Declared in 15 sources; zero hits in the package source of **both** 0.1.5 and 0.1.7; hidden state initialised unconditionally by `init_hTtime()` returning `torch.zeros(...)`; `HydraNetConfig.model_config` is `extra="allow"`, so it validates silently. **The one that holds is the one where a grep for the key as a READ was actually run.** ⚠️ **The shape of the error, which is what generalises: all three subjects look identical from the declaration side, and the distinguishing evidence is always in the consumer.** This is **C-157**'s finding — a dead key, a missing key and a look-alike key are indistinguishable by reading — restated as a *procedural* failure rather than a schema one: C-157 says the configs cannot tell you, and this entry says the habit is to answer anyway. ⛔ **Why Tier 3 and not Tier 4:** the output of this procedure is a confident statement to the operator, and the operator acts on it. Two of three were wrong and one of those cost a 300-lesson GPU run and ~$8 of pod time, plus the five hours of re-planning around it. ✅ **The check is one command and it is not optional:** `grep -rn '\["<key>"\]\|\.get("<key>"\|config\.<key>' <package source>` over the version that actually ran — the *read*, never the declaration. If the grep is not run, the honest answer is "I do not know whether anything reads it", which is C-160's rule ("nothing matched" ≠ "I could not see") applied to one's own diagnosis. Cross-refs: **C-68** (the `config_meta` mechanism, inverted by this session), **C-157** (dead/missing/look-alike keys are indistinguishable by reading — the structural root), **views-hydranet C-324** (the inert-knob signature: `extra="allow"` validates a key that nothing consumes), **C-176** (the same habit applied to a telemetry field rather than a config key), **C-104** (CI validates a key the runtime may not read), **C-168** (ROOT). Member of **Cluster G** (a check out of scope returns a clean verdict). |

---

### C-176 — A blank telemetry field is reported as a gap without checking whether blankness is the designed signal

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | Reporting an empty column, an unset field or a missing file as missing data — especially in a delivered artefact, and especially to the team that owns the field. |
| **Source** | Self-inflicted 2026-10-25; corrected by the views-hydranet session the same day and verified against the wheel that actually ran |
| **Status** | Open |
| **Location** | The reporting procedure, not a file. The subject it was wrong about: `views_hydranet/train/training_engine.py:1663-1673` (the writer that deliberately starts `fed_max` at `None`), pinned by `tests/test_trajectory_log_end_to_end.py:236`. The artefact read: `$OUT/logs/trajectory.csv`. |
| **Notes** | **`trajectory.csv`'s `fed_grad_max` is empty in all 300 rows of the control run. This seat flagged that to the owning session as a defect, with the reasoning that it "will read as a measurement rather than a gap". That is exactly backwards, and the inversion is the whole design.** ⛔ **Verified in the 0.1.7 wheel that trained the run**, `views_hydranet/train/training_engine.py:1663-1673`: *"`fed_max` starts at None, NOT 0.0. … A 0.0 initialiser therefore wrote a real-looking zero on every production run, making 'the wire is off' and 'the wire is on and the gradient died' the same three characters in the CSV. … `None` ⇒ blank cell ⇒ `read_csv` gives NaN ⇒ 'not measured', which is the truth."* The column measures gradient through the autoregressive feedback wire, which is fed only when `ss_backprop_through_feedback` is True; that defaults **False** (`config_initializer.py:291`) and no roster config sets it. So nothing was measured and the column says nothing — correctly. Pinned by `tests/test_trajectory_log_end_to_end.py:236` (`test_fed_grad_max_is_BLANK_not_zero_when_the_feedback_wire_is_off`). **A populated `fed_grad_max` would be the anomaly.** ⚠️ **Tier 4 because no output was wrong and nothing was acted on** — but the advice, if taken, would have reversed a deliberate fix and reintroduced the exact ambiguity its author had removed, in a delivered artefact, having been argued for on the authority of a 300-row observation. ⚠️ **The generalisable rule, and it is one this repository already holds from the other side:** `feedback_absence_is_not_a_pass` says a check that cannot read its input must FAIL rather than pass. The corollary is that **a well-built instrument distinguishes "not measured" from "measured as zero", and blankness is how it does it** — so an empty field is as likely to be good engineering as a gap, and the only way to tell is to read the writer. **Before reporting an absence, read the code that would have written the value.** Cross-refs: **C-175** (the same habit applied to a config key — a default read as proof of deadness), **C-159**/**C-160** (absence is not a pass: the same distinction, with the polarity the other way up), **C-157** (a missing key and a dead key are indistinguishable by reading). Member of **Cluster G** (a check out of scope returns a clean verdict). |

---

### C-177 — Three shell idioms that misreport or self-harm during remote pod operation: self-matching `pkill`, a backgrounded job that holds the SSH channel, and `read` returning non-zero on success

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | Writing or reusing a monitor, launcher or kill command that runs over SSH against a live pod — particularly one that pattern-matches process names, backgrounds a long-running script, or parses a probe's output with `read`. |
| **Source** | Seven instances across one session, 2026-10-25 (`heavy_freighter` rehearsal and control run) |
| **Status** | Open |
| **Notes** | Three distinct mechanisms, one shared root: **a shell construct whose failure mode is a misleading report, or self-harm, rather than an error.** ⛔ **(1) `pkill -f <script>.sh` kills the invoking shell. Three instances.** `-f` matches the full command line, and the SSH command line *contains the pattern it is searching for*, so the pattern matches itself. First instance produced no output at all; the next two exited **144**. Fixes, in increasing order of reliability: bracket the pattern so it cannot match itself (`pod_run[_]model`, `guard_artifac[t]`), keep the literal filename out of every other part of the same command, and — what finally worked — **do not kill at all**: write the replacement under a new filename and leave the old process to finish. ⛔ **(2) `setsid nohup ./script > log 2>&1 & sleep 8` holds the SSH channel open for the script's whole lifetime.** Observed as a 120-second hang on a command expected to return in 8. Backgrounding detaches the job from the shell but not the inherited file descriptors from the channel. Working form: `ssh -n host '(setsid ./script > /log 2>&1 < /dev/null &) ; echo launched'` — subshell, explicit `< /dev/null`, `-n` on the client, and no `sleep`. ⛔ **(3) `printf '%s %s %s'` with no trailing newline makes a subsequent `read` return non-zero *even though it read the data*, so the error branch fires on success.** A monitor reported **"ssh failed — pod gone?"** while the probe was in fact working perfectly, during a live paid run — the most alarming possible false alarm, and the one that invites killing a healthy run. A `sed` patch then left ambiguous `\n` quoting across three layers of nesting (laptop shell → ssh → pod shell); the fix was to stop fighting it and restructure: `$(...)` capture plus `IFS='|' read -r ... <<< "$R"`. ⚠️ **Tier 4 — none of these corrupted data or a result.** They are registered because they were self-inflicted **seven times in one session while supervising a paid 2h40m GPU run**, and because in each case the symptom was a *confident wrong signal* rather than a failure: a silent kill, an apparent hang, and a false "pod gone". Under time pressure on rented hardware, a confident wrong signal is what provokes a destructive response. ✅ **The common precaution:** in remote-operation scripts, treat every "no output", "non-zero exit" and "pattern matched" as needing a positive control before it is believed — and never let a command's own text be inside its own search space. Cross-refs: **C-169** (an empty result from a broken sweep read as a clean answer — same family, higher stakes), **C-171** (a monitor that reported the wrong run's progress), **C-159**/**C-160** (absence is not a pass). **review-rr (2026-10-25) — flagged for split, not split here:** this entry carries three unrelated idioms under one trigger, so it fires for all three or none. The third (`printf` without a trailing newline making `read` return non-zero on success) can produce a false "ssh failed — pod gone?" and therefore a wrong decision about a live host; the other two are self-harm that announces itself. Splitting creates new IDs, which a review pass may not do — register the third separately when it next fires. No cluster: operational idioms, not a structural root. |

---

### C-178 — `git switch -c <branch> origin/development` sets the new branch's push target to `origin/development`, so a routine `git push` targets the protected integration branch

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Creating a feature branch with an explicit start point (`git switch -c <new> origin/<base>` or `git checkout -b <new> origin/<base>`) and later running a bare `git push` on it. |
| **Source** | Recurrence caught 2026-10-25 while branching for the diagnostic-viz change; previously encountered in the same form |
| **Status** | Open |
| **Location** | The branching step of the eleven-step merge ritual; `CLAUDE.md` (integrate on `development`, never target `main`); project memory `project_branching_development_main.md` |
| **Notes** | **Naming a remote-tracking branch as the start point makes git set it as the new branch's upstream.** A bare `git push` on that branch then pushes to `origin/development` rather than creating `origin/<branch>` — and Simon's standing instruction is that `development` is **never** pushed directly, because `update_catalogs.yml` commits back to it after a merge (C-112's neighbourhood) and because every feature PR is supposed to target it rather than originate in it. ⛔ **Why it is more than a typo:** the hazardous state is created by the *correct* command. Branching from `origin/development` rather than a possibly-stale local `development` is the right thing to do — the shared checkout drifts durably behind the remote (**C-105**) — so the safe habit and the dangerous side effect arrive together, and nothing warns. The branch looks normal in `git branch`; the upstream is visible only to `git branch -vv` or `git status -sb`, neither of which is part of the ritual. ✅ **The fix is one command, immediately after creating the branch:** `git branch --unset-upstream`, then verify with `git status -sb` that the branch reports no upstream (so the first push must be explicit: `git push -u origin <branch>`). Applied on 2026-10-25 before any push occurred. ⚠️ **Registered rather than left as operator discipline** because the standing rule it endangers ("never push `development` directly") is one whose violation is effectively irreversible on a shared integration branch, and because the ritual that governs branching does not currently contain the check. Cross-refs: **C-105** (the shared checkout drifts behind `origin/development` — the reason the hazardous form is also the correct form), **C-112** (behaviour changes riding on the wrong branch/release), **C-168** (ROOT, loosely: a state believed from the command issued rather than from the state read back). No cluster — standalone operational hazard. |

---
### C-179 — A retrieval's scope parameter silently becomes the verification's scope, so a sync that covers one of two output directories is certified complete by a count taken inside the one it covered

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Writing any sync, rsync, watcher or retrieval that names a timestamped run directory — `diagnostics/<stamp>/`, `offline-run-<stamp>/`, `predictions_<stamp>/` — as a literal path, and then reporting completeness by comparing counts between source and destination. |
| **Source** | `/falsify` on "I can terminate the pod", 2026-10-26, probe P3 (Category F, adversarial — attacking the auditor's own exclusion list) during the `heavy_freighter` #607 structure-penalty arm run |
| **Status** | Open |
| **Location** | The retrieval procedure, not a file. The subject: `models/<model>/reports/plots/diagnostics/<YYMMDD_HHMM>/`, which HydraNet creates **once per pipeline stage**, not once per run — `261026_0143` at train start, `261026_0410` at evaluate start. Nearest code: `views_hydranet/utils/visual_diagnostics.py` (the writer); `docs/runpod_run_guide.md` Phase 5 (the rsync). |
| **Notes** | **A live plot feed was armed at 03:56 against `diagnostics/261026_0143/` — the only run folder that existed at that moment. The evaluate stage created `261026_0410/` at 04:10 and wrote 17 files into it, including all 13 per-origin `03_model_reasoning` biopsies, which are the forecast-reasoning plots and the most readout-relevant images the run produces.** ⛔ **The verification could not have caught it, and said so in the affirmative.** Completeness was reported as *"1812 png, 900 npz, both sides"* — a parity check whose two numbers were **both counted inside the hardcoded directory**. The sync's scope parameter had become the audit's scope parameter, so the instrument agreed with itself exactly as C-168 describes, and the agreement was rendered as a pass. The gap surfaced only when a probe counted `reports/plots` **as a whole**: pod **2729**, laptop **2712**. ⛔ **The evidence was already in this register and in this session.** C-168's own Notes record the control run's plot diff as *"**2729/2729**, both diagnostics timestamp dirs"* — the two-folder shape, written down, with the same file count. The control's folders (`251026_1043` + `251026_1321`) had additionally been **printed in this session's own output** before the feed was written. A one-folder sync was then written anyway. ✅ **The fix is to enumerate, not to name:** list `diagnostics/*/` on both sides, assert the **set of folder names** is equal and non-empty, and only then compare contents. A count comparison must never be rendered as verified in any case — it cannot detect a truncated transfer, and rsync's default mtime+size quick-check can skip a file rewritten within the same second at the same length. ✅ **Remediated 2026-10-26:** 17 files retrieved, and all **2729/2729** plot files re-verified by per-file sha256 across both folders. Had the pod been destroyed on the preceding "safe to terminate" statement, the 13 biopsies were not reproducible (**views-hydranet C-317**). Cross-refs: **C-168** (ROOT — completeness verified against the producer's own declared scope; this is that error with the scope set by the *consumer's* own command line rather than the producer's directory), **C-169** (the sweep's own blind spot, same session class), **C-170** (the classes outside `$OUT`), **C-180** (the digest comparison in the same audit that returned empty and read as agreement), **C-171** (Cluster H — identity of a file against *this* run), **views-hydranet C-317** (not reproducible, so a lost plot is lost). Member of **Cluster G** (a check out of scope returns a clean verdict). |

---

### C-180 — A digest comparison whose digests are empty reports agreement, because the filenames match and nothing asserts that a hash was computed

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Comparing two checksum manifests with `diff`, `comm` or `join` — especially when either side is produced by a shell pipeline (`tar --to-command`, `xargs sha256sum`, `find -exec`) whose quoting can fail without a non-zero exit. |
| **Source** | `/falsify` on "I can terminate the pod", 2026-10-26, probe P2 (Category E, consistency) — the probe's own instrument failing mid-probe |
| **Status** | Open |
| **Location** | The verification command, not a file. The construct that failed: `tar -xf - --to-command='sh -c "...sha256sum..."'` with nested quoting, whose inner command substitution never received stdin. |
| **Notes** | **Verifying the 78 members inside a 683 MB gate-posterior archive, the local side emitted a filename for every member and an EMPTY string for every hash. The comparison then reported no differences in all three buckets — only-on-source, only-on-destination, and differing — because the FILENAMES matched perfectly on both sides. The result was shaped exactly like a pass.** ⛔ **It was caught by accident.** `diff -q` disagreed with `comm`, which prompted printing the raw line through `cat -A`; the hash column was visibly absent. Nothing in the comparison itself would ever have reported it: `comm` on an empty-vs-empty value set is silent by construction, and the pipeline exited 0. ⛔ **This is C-168's rule occurring inside a probe written to detect C-168, in an audit of a claim that rests entirely on verification — the third consecutive layer at which the instrument rather than the subject was the defect** (C-159 the mutation harness, C-169 the omission sweep, this entry the digest comparison). ✅ **The fix is a positive control on the instrument, not on the subject, which is ADR-025 §2.4 applied one level in:** before comparing, assert the digest count is non-zero, assert every digest is 64 hex characters, and assert no member is zero-length. The corrected probe (Python, `assert n > 0` and `assert empty == 0`) printed *"hashed 78 members, none zero-length"* and *"local hashes are non-empty: 78/78"* **before** the comparison was allowed to render a verdict. ⚠️ **Generalisable:** any comparison keyed on a stable column (path, id, name) while the informative column (hash, size, value) is silently absent will report agreement. The key matching is not evidence that the value was read. Cross-refs: **C-168** (ROOT), **C-159** (a mutation that silently fails to apply reads as a result — same "instrument's own input unverified" root), **C-160** (a check that cannot read its input reports a clean answer), **C-169** (the sweep that returned empty and read as clean), **C-179** (the other falsification from the same audit), **ADR-025** §2.4 (the independent-control requirement this violates). Member of **Cluster G** (a check out of scope returns a clean verdict). |

---

### C-181 — The sweep reports a mount it cannot write to as UNVERIFIED, so 56 single-file bind mounts produce a refusal indistinguishable from a genuinely opaque filesystem

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Reading `pod_sweep_outputs.sh`'s exit 1 before a teardown, on any container host whose runtime bind-mounts individual files — every RunPod/Docker GPU image does, for the NVIDIA driver libraries and `/etc/hosts`. |
| **Source** | `/falsify` on "I can terminate the pod", 2026-10-26 — observed running the sweep on the `heavy_freighter` #607 arm pod |
| **Status** | Open |
| **Location** | `tools/podrun/pod_sweep_outputs.sh:117-126` (the sentinel plant, and the `cannot-plant-sentinel` branch), on PR #589 / branch `docs/572-sigma-hat-version-null`; not on `development`. |
| **Notes** | **The sweep plants a sentinel on each mount and requires finding it before an empty result from that mount may mean anything — the right design. But a mount point that is a regular file cannot hold a sentinel and cannot hold run output either, and the script maps both conditions to the same verdict.** ⛔ **Measured:** the arm run's sweep exited **1** with **56 UNVERIFIED mounts**, every one of them a single-file bind mount — `libcuda.so.595.91.07`, `nvidia-smi`, `/etc/hosts`, `/etc/resolv.conf`. The one mount that could hold output, `/workspace`, was verified and reported 44,303 touched files. So the refusal was correct in form and entirely benign in substance. ⚠️ **Why this is a defect and not a cosmetic complaint.** The contract is *exit 1 → DO NOT DESTROY*. A verdict that is unverified for 56 structurally benign reasons on every single run trains the reader to look past exit 1 to the reason, and then to wave it through — which converts the guard into decoration by exactly the route C-106 and #602 describe. **A guard whose refusal is always benign stops being read**, and the one time the refusal is not benign is the run it exists for. ✅ **The fix is a third verdict on the mount, not a fourth on the sweep:** classify a mount point that is not a directory as **EXCLUDED-WITH-REASON** (a regular file cannot contain files), keep **UNVERIFIED** for mounts that are directories the sentinel genuinely could not reach, and let exit 1 fire only on the latter. ⚠️ Found alongside the `overlay` blind spot recorded as a second instance at **C-169**; both are defects in the tool written to remedy C-169, found on its first use against a real teardown. Cross-refs: **C-169** (same tool, the exclusion that hid the root filesystem), **C-168** (ROOT), **C-106** (a guard off the execution path is not a guard), **ADR-025** (three verdicts, unverified dominates — this entry is about keeping UNVERIFIED meaningful enough to dominate). Member of **Cluster G** (a check out of scope returns a clean verdict). |


---

### C-182 — A guard that discovers its subject by matching source text, or asserts a cardinality where the property is an identity, stops applying without going red

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Writing or **widening** a guard that finds its subject by matching source text — `re.search` over a config, `"<name>" in main_text`, a grep for a class name, `inspect` over a decorator — or that asserts a **cardinality** (`len(argvalues)`, a tenant count, `sorted(floors.values())`) where the property it defends is an **identity**. Also: reading a mutation set's own kill rate as a coverage claim. |
| **Source** | Independent guard-mode `/falsify` audit of `09a3992b` (epic #595 S3, PR #634), 2026-10-27 — clean-context subagent, 24 mutations applied and executed |
| **Status** | Mitigated |
| **Location** | Fixed in `e39362b7`: `tests/test_chrome_box_pairing.py`, `tests/test_ensemble_configs.py`, `tests/test_delivery_map_truth.py`, `tests/test_environment_sharing.py`, `tests/test_requirements_hygiene.py`, `tests/test_pfe_production_readiness.py::_discover_pfe_ensembles` |
| **Notes** | **The distribution is the finding, not any one guard.** 24 mutations, 12 caught, 12 survived, and the split is clean along one axis: **every mutation to a VALUE inside a field a guard reads was caught (12/12); every mutation to how a guard decides WHAT to read survived (12/12).** Window months, lambda_mix, level, member order, duplicates, source markers, aggregation, one-sided and both-sided edits — all caught, impressively. Quote style, a `#` comment, statement order, a key enumeration, `parametrize` length-vs-content, a `pytest.mark.skip`, tenant identity-vs-count, floor label-vs-count, and a manager class name used as a discovery token — all survived. ⛔ **Two of these are self-refuting and name the mechanism.** (a) Two guards had *already* been widened in response to an earlier mutation proof, and both widenings were the same mechanism one notch tighter — `"build_reconciler" in text` → a regex on text; `name in text` → a longer string in text. The next mutation walked up the same ladder (`reconciler = None` with the build call left as a comment; a reformatted config key). **Pinning a tighter string is not a different kind of guard.** (b) A meta-guard existed *specifically* because narrowing a `parametrize` had been a surviving mutation in this repository before, and it pinned `len(argvalues) == 8` — so seven distinct pairs padded to eight, and a skip mark on one pair, both passed it. A guard against a class of defect, open to that class of defect. ⛔ **The worst instance was a discovery token held in a comment.** `test_pfe_production_readiness.py::_discover_pfe_ensembles()` found its subjects by grepping `main.py` for `PredictionFrameEnsembleManager`, so switching `chrome_box` to the DataFrame `EnsembleManager` — which **refuses** its `pgm_cm` reconciliation and would publish unreconciled predictions — *plus deleting the ⛔ comment that warns against doing so* left all 10,403 tests green: the ensemble simply left the census, five cases stopped being collected, and **nothing turned red**. A smaller census reports fewer passes, not a failure. ✅ **Exits, all applied:** read the config by **loading** it, never by matching its text; read `main.py` through **`ast`**, so a comment has no node and statement **order** is visible (the reconciler build must lexically precede the call it is passed to); compare a `parametrize`'s **argvalues to the declared table element-wise** and reject any `marks`; compare **every shared key** between two config copies rather than a hand-written list of two; and replace cardinality assertions with **identity** ones (`floors == {specifier: count}` keyed, the environment's **tenants by name**). ⚠️ **One correction worth preserving: the audit's own suggested fix for the floor census was wrong, and only re-running the mutation found that out.** It said to key the count dict by specifier instead of sorting the values; that pins which *specifier* holds which count, while the mutation moves which *file* holds which floor, so it survived unchanged. The working fix pins the nineteen load-bearing declarations **by name**. A fix for a surviving mutation is a hypothesis until the mutation is re-run against it. ⚠️ **And the meta-lesson about proof:** the author's own set was 19 mutations, 19 killed — accurate about what it ran, misleading about what it covered, because all 19 were value mutations. **A 19/19 kill rate inside one mutation set is not a coverage claim**, since the set is written by the model of failure that wrote the guards. This is why `ship-it` 5.5 requires a clean-context auditor, and it is the second consecutive commit where an independent audit found survivors the author's pass did not (#601: 13 survivors in 20 guards, sharing one blind spot — every guard injected the same stub, so nothing pinned what the provider asked for). **Residual (why Mitigated, not Resolved):** a compensating swap between two environments *other* than `views_ensemble` is still invisible to `test_environment_sharing.py` — converting all ~180 launchers to named sets is a fleet-wide change and was out of S3's scope. Cross-refs: **C-127** (ROOT of the text-matching half — a double-quote-only pattern matched 47 of 128 files, and the guard fixed here cited C-127 four lines above its own defect), **C-161** and **C-180** and **C-159** (the cardinality half — a count accepted as proof of parity), **C-179** (a check whose scope is set by the command that wrote it), **C-113** (a gate that cannot tell "clean" from "did not run"), **C-92**/**C-106** (a skip silently removing coverage), **C-164** (by-hand mutation passes). Member of **Cluster G** (a check out of scope returns a clean verdict), and governed by **ADR-025** — this is ADR-025's sibling failure: not a check that *cannot see* its subject and passes anyway, but one that *silently stops looking* and reports fewer passes.

---

### C-183 — The PFE forecast path holds every constituent's frames AND the pooled frames at once, so a 1,024-draw global ensemble peaks at ~2x its pool and the `RAM >= 50 GB` pod rule does not cover it

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Renting a pod for, or launching, any `prediction_format: "prediction_frame"` ensemble pooling more than ~256 draws at global land — `chrome_box`/`chrome_crate` today. Also raising `expected_samples_per_model`, adding a constituent, or adding a regression target on any such ensemble. |
| **Source** | Arithmetic traced through pipeline-core 3.3.0 during epic #595 S3 (#598), 2026-10-27, after the maintainer challenged a hand-waved memory claim: *"that's not a problem on the pods we usually go for is it? you tell me you actually check this. All the numbers are here."* |
| **Status** | Open |
| **Location** | `views_pipeline_core/managers/ensemble/prediction_frame_ensemble.py::_forecast_ensemble` — `model_results` is built for every constituent and every target **before** the aggregation loop (`:633-640`) and freed nowhere in the method; `forecasts[target] = agg_pf` (`:752`) accumulates the pooled frames beside it. Consumers: `ensembles/{chrome_box,chrome_crate}/configs/config_hyperparameters.py`. The rule it exceeds: `docs/runpod_run_guide.md` Phase 1.1. |
| **Notes** | **This is C-155's own carry-forward, confirmed one tier up.** C-155 closes by predicting it: *"the memmap path is cheaper per unit but has the same shape — instances also accumulate in a list and are freed only after the loop. Re-measure before restoring any sample count."* The ensemble path on the PredictionFrame tier has that shape, and nothing re-measured until now. ⛔ **Peak is roughly double the pool, and the pool is the number everyone quotes.** For `chrome_box` at 64,742 cells x 36 steps x 1,024 draws x 3 targets, float32: `model_results` (8 members x 3 targets x 128 draws) = **26.67 GiB**, `forecasts` (3 targets x 1,024 pooled draws) = **26.67 GiB**, input volume 0.43 GiB, cm frame 0.08 GiB — **peak ~53.85 GiB**, against a selection rule of **>= 50 GiB** and the two machines the 2026-09 campaign actually used, 57 GiB (94% — not a fit) and 87 GiB (62% — the class to rent). ⚠️ **94% is not "fits", and this register already holds the evidence:** the 31 GB machine the runpod postmortem says *"do not take"* collapsed **25x** (32 -> 0.11 steps/s) by *approaching* its ceiling at ~20 of 31 GB through kernel reclaim — nothing failed, it only slowed, for eight hours. Headroom is the property, not fit. ⚠️ `chunk_by_time=True` (C-200a) bounds the **reconciliation's** working set and does nothing for either term above, which is why citing it as the answer is wrong. ⚠️ **Unit trap at the boundary:** `rusty_bucket`'s *"~28.6 GB"* is decimal and 26.67 GiB is the same quantity; the run guide's own check is `memory.max / 1073741824`, i.e. GiB, so the figures here are GiB and comparable to the rule. A 7% confusion sits exactly where the fit/no-fit line is. ⛔ **What is NOT measured:** transient allocations inside `np.concatenate` and inside the reconciler beyond the frames counted. So ~53.85 GiB is a **floor on the peak, not a ceiling** — a further reason for the 87 GiB class. **The upstream issue already exists and does not yet know about this path: views-pipeline-core#463**, *"the PFE **evaluation** loop retains every aggregated frame — are there low-hanging fruits before we talk about streaming?"* — open, and scoped to `_evaluate_ensemble`. The **forecast** path is the worse case and is not in its scope: `_evaluate_ensemble` retains the aggregated frames, while `_forecast_ensemble` retains the aggregated frames **and** every constituent's, which is what doubles the peak. ⚠️ Commenting the measurement onto #463 is another repository and therefore the maintainer's call, not this seat's; the figures are in this entry so the link survives either way. **Exits, in order of value:** (1) free each constituent's frames as they are consumed, or stream the aggregation per target rather than building `model_results` whole — the upstream fix, tracked at views-pipeline-core#463, and the one that makes the >= 50 GB rule true again; (2) failing that, raise the pod rule *for this class of ensemble* in `docs/runpod_run_guide.md` rather than leaving one global number that is right for a single model and wrong for a pooled ensemble; (3) a preflight that computes this arithmetic from the configs and refuses, since **the in-code memory guard reads the HOST's RAM and will approve a run that cannot fit** (run guide, Phase 5) — the same defect class as C-112. ⚠️ **How this was nearly missed, recorded because the pattern is the point:** the figure was first raised as a blocker needing the maintainer's decision (wrong — the pool fits a pod), then as *"fits the >= 50 GB rule with room"* (also wrong — the peak is double the pool). **Two confident verdicts in opposite directions, neither measured**, which is precisely the failure C-168 and the σ̂ readout record. The selection rule, the postmortem and C-155 were all in this repo the whole time. Cross-refs: **C-155** (ROOT — same accumulate-then-release shape, `dataframe` tier, 303 GB at 100 samples; its carry-forward note is this entry), **C-99** (*"production hardware"* is the laptop rotation — the scope confusion that started this), **C-112** (a gate deciding availability in a scope that cannot answer it), **C-116** (shared environments), **C-168** (a verdict believed from the instrument's own scope), **views-pipeline-core#463** (the upstream home, evaluation-scoped), **views-pipeline-core#181/#189** (the host-RAM OOM in the eval tail — same family, different stage). No cluster — a resource-shape defect, not a blind check. |

---

### C-184 — The pipeline-core floor is guarded over the 31 directories that declare it, but the thing it protects is a shared conda prefix with 45 tenants, so 29 unwatched tenants can pull the whole environment below the floor

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Adding or editing a `requirements.txt` under `models/` for any tenant of `envs/views-baseline` that is **not** one of the sixteen `*_sparrow`/`*_condor` members — in particular adding a `views-pipeline-core` or `views_pipeline_core` line to one. Also: any future change that makes a non-census tenant of a census prefix install pipeline-core, including through a transitive dependency's own floor. |
| **Source** | Independent guard-mode `/falsify` on #635's commits (`34c1aa0d`, `d4cede65`), 2026-10-27, run in a clean-context subagent per ship-it 5.5. 25 mutations, 9 survivors; this is the pattern the survivors shared, and it is the auditor's own framing rather than the author's. |
| **Status** | Open |
| **Location** | `tests/test_requirements_hygiene.py::test_every_pipeline_core_declaration_is_at_the_floor_its_source_needs` — its census is `DECLARES_PIPELINE_CORE`, 31 named directories. The prefix each tenant installs into is in that tenant's `run.sh` `env_path`, which this guard never reads. Measured 2026-10-27: `envs/views-baseline` has **45** tenants, **16** of them in the census, **29** outside it (the four `*_baseline`/`*_cmbaseline`/`*_pgmbaseline` families, eight `*_ranger`, nine `*_dwarf`, six `*_dream`, two `*_strider`). |
| **Notes** | ⛔ **The guard was hardened against the attack that had already happened, and the survivors all attacked a different axis.** C-182 was a compensating *count* swap, and #635's identity assert genuinely closes it — proven by mutation (`ashen_sparrow`'s declaration moved to `chrome_box`, counts held at 31 and one specifier key, guard went red). But #594 is not a property of 31 directories. It is a property of **which pipeline-core version lands in a conda prefix**, and C-116 is already in this register saying exactly that: *"131 files resolve into 11 shared environments… whichever ran last wins."* The module's own docstring cites C-116 as the reason inheriting a co-tenant's floor is not good enough — and then the guard built on that reasoning watches the dependants instead of the prefix. ✅ **Four narrower survivors from the same audit were fixed inside #635 and each proved by re-running its own mutation:** (a) `packaging` does **not** canonicalise `Requirement.name`, so `views_pipeline_core==3.1.0` was a different string to the guard and the same package to pip — closed with `canonicalize_name`; (b) `name @ git+https://…` set `req.url`, and the guard's `or req.url: continue` dropped the whole directory out of the census rather than flagging it — a smaller census reports fewer passes, C-182's other half; (c) `views-pipeline-core>=3.2.0,<4.0.0; platform_system == "Java"` left `str(req.specifier)` untouched while installing nothing; (d) renaming `requirements.txt` to `base_requirements.txt` kept the directory in the census via `Path(name).parent` while pip stopped reading the file. All four now fail loudly: a declaration that cannot be read as a floor makes the guard **refuse** rather than skip (ADR-025). ⚠️ **What is NOT fixed, and why it was not fixed here:** asserting the census over environment co-tenants means parsing `env_path` out of ~180 `run.sh` files, and `run.sh` is production infrastructure this session may not modify while nine of the 29 unwatched tenants are `*_dwarf` directories another session owns. That is a fleet-wide change, not a line in #635. **Mitigating fact, measured rather than assumed: none of the 29 declares pipeline-core today**, so the surface is real but currently empty — one added line opens it, which is what makes the trigger above specific rather than perpetual. ⚠️ **Second residual:** `DEFERRED_PACKAGES` can be given a `views-pipeline-core` entry again at any time, which silently removes the package from `test_no_dependency_is_declared_without_an_upper_bound` and `test_a_package_is_declared_the_same_way_everywhere`; `test_the_deferred_list_stays_small_enough_to_be_honest` permits two entries and would stay green. ⚠️ **One audit finding rejected after checking it, recorded so the rejection is reviewable:** the auditor called `>=3.2,<4.0.0` and `==3.3.0` false positives because neither admits anything below 3.2.0. Measured: both fire **two** guards, the second being `test_a_package_is_declared_the_same_way_everywhere`, so a lone variant spelling is red by the repo's general rule and not by this guard's brittleness. A *uniform* bump of all 31 to `==3.3.0` does fire the floor guard alone — that is a real brittleness, and it is a 31-file reviewed edit by construction, so one line of test churn is the correct cost. Cross-refs: **C-116** (ROOT: shared environments, last writer wins — this entry is C-116 reaching a guard that cites it), **C-182** (the axis that *was* closed, and the "smaller census reports fewer passes" mechanism), **C-127** (text matching as discovery), **C-112** (a guard reading the wrong host), **#594** (the incident), **#635** (where this was found). Member of **Cluster G** (a check out of scope returns a clean verdict), governed by **ADR-025**. |

---

### C-185 — A prefix was measured THROUGH a symlink, so the target's facts were reported as the link's, and the repair nearly ran on the wrong environment

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Reading `conda-meta/history`, `bin/python`, `site-packages` or any other file *inside* a path under `envs/`, `data/`, or a pod mount, in order to characterise that path — without first establishing with `ls -ld` / `[ -L ]` whether the path is the object or a pointer to one. Also: writing an `rm -rf <path>/` remedy for anything under `envs/`. |
| **Source** | Self-reported during #648, after the maintainer challenged the diagnosis (2026-10-30) |
| **Status** | Resolved (the diagnosis; the class remains) |
| **Location** | `envs/views-baseline` and `envs/views-hydranet`, both symlinks into `~/anaconda3/envs/` created 23–24 May 2026 and removed 2026-10-30. The mis-measurement was reported to the maintainer and written into a memory file before being caught. |
| **Notes** | The prefixes were reported as **copies** of two personal anaconda environments, on two pieces of evidence: each one's `conda-meta/history` first line reads `conda create -n views_pipeline` rather than `--prefix`, and `bin/python3.11` is a real 24,666,920-byte binary rather than a link. Both observations were correct and both were facts **about the other end of the symlink** — every read followed the pointer, and nothing in the output says so. ⛔ **Two concrete harms, one avoided by luck.** (a) The wrong diagnosis implies the wrong repair: a copy means rewriting 54 + 67 baked absolute shebangs, a symlink means one `rm`, and those are different work items on different people's plates. (b) The remedy drafted from it was `rm -rf <path>`; written with a trailing slash — the form a hurrying operator reaches for — it would have **emptied the maintainer's real `views_pipeline` environment, the one this repository's test suite runs in.** The destructive case and the diagnostic case have the same root: a path that follows a link is not the object you named. **Exit:** establish identity before measuring (`ls -ld`, `[ -L ]`, `realpath`), and for anything under `envs/` state in the remedy whether it unlinks a pointer or deletes a tree. Distinct from **C-112** in mechanism — that is shell scope versus exported scope — but the same badly-posed question C-112's own closing line names: *a check must be asked in the context it will answer for.* Cross-refs: **C-112**, **C-149** (the prefix nobody checks), **C-184** (the prefix as the real subject), **#643**, **#644**. Member of **Cluster G** (a check out of scope returns a clean verdict). |

---

### C-186 — A delivery declaration recorded an upload as having never happened, while the run log proving it sat two directories away

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Deciding what a partner has already received — to re-arm a delivery, to answer an external party, or to assess exposure to a sourcing rule — from the prose history in `deliveries/<consumer>.py` or a postprocessor README, rather than from `postprocessors/<consumer>/wandb/run-*/files/output.log` and the store itself. |
| **Source** | `/code-review medium` on PR #648 (2026-10-30), verified against the log before registering |
| **Status** | Mitigated (both files corrected in #648; the class and the shelf state remain) |
| **Location** | `deliveries/un_crafd.py` (the "Why it is no longer `rusty_bucket`" paragraph) and `postprocessors/un_crafd/README.md`. Evidence: `postprocessors/un_crafd/wandb/run-20260814_202751-g06spd7j/files/output.log`; the writers are `views_postprocessing/contract/wire/sink.py:165` and `crafd/managers/crafd.py:396`. |
| **Notes** | Both files stated that nothing had ever been delivered to CRAF'd — *"Nothing had fired only because `monthly_run.sh:179` runs `un_fao` and not this one; that is luck, not a safeguard"* and *"CRAF'd has never received either"*. The log records **111 uploads completing 2026-08-14 20:36:25**: 108 shards (36 months × `lr_ged_sb`, `lr_ged_ns`, `lr_ged_os`), `rusty_bucket_forecasting_20260727_095355__sidecar.parquet`, the `__manifest.json` ADR-013 commit marker, and `historical_dataset_20260814_203554.parquet`. An earlier attempt the same morning (`run-20260814_040036-az5srom5`) failed on an Appwrite schema error. ⛔ **So `rusty_bucket` — FAO's ensemble — reached CRAF'd two months before the 2026-10-22 ruling that it must not**, and the decision file said the opposite. **Why it survived:** the contradicting evidence was never hidden. `tools/liveness/crafd_delivery.py`'s own docstring says *"verified live 2026-08-24 … 111 files"* — the same delivery counted from the other end — and the two statements sat in one repository for two months without meeting, because one is prose in a declaration file and the other is a count in a liveness module. Nobody greps a `wandb/` directory to check a sentence. **This is C-126 inverted and is registered separately for that reason:** C-126 is *execution that happened in no record*; this is *a record asserting non-execution*. The fix for C-126 — derived status reported beside declared intent (ADR-017 §7) — would have caught this too, which raises its value. ⚠️ **What to do about shards already on the partner's shelf is the maintainer's decision** and is not recorded as resolved here. Cross-refs: **C-126**, **C-121** (unbounded age at the boundary), **C-97** (delivery by recency, not identity), **C-144** (`check()` has no runtime caller), **C-135**, **C-139**. |

---

### C-187 — An evidence directory contained a module CPython auto-imports, so reading the evidence could silently patch the code under test

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Adding a file to `tools/podrun/evidence/` (or any retained artefact directory) whose name CPython treats as special — `sitecustomize.py`, `usercustomize.py`, `conftest.py`, or the name of any module the repository imports — and then starting an interpreter with that directory as the working directory. |
| **Source** | `/code-review medium` on PR #648 (2026-10-30) |
| **Status** | Resolved (renamed `probe_sitecustomize.py` in #648, contents byte-identical, original name recorded in the directory's README) |
| **Location** | `tools/podrun/evidence/2026-10-28_chrome_box/probe_sitecustomize.py`, formerly `sitecustomize.py` |
| **Notes** | CPython imports any module named `sitecustomize` found on `sys.path` at interpreter start, and `sys.path` includes the working directory. The file monkeypatches `prediction_frame_ensemble.load_cm_frame` and `DatafactoryCountryMappingProvider.build` and inserts `/workspace/views-models` into `sys.path` — which is exactly what it was written to do **on the pod**, where the name is what made it load. Retained as evidence in-repo, the same name means that anyone who `cd`s into the directory to read the artefacts and runs `python` gets the patches, silently, in a repository where both patched functions are under active change. ⚠️ **Aggravated by the same PR's `[tool.ruff] extend-exclude = ["tools/podrun/evidence"]`**, added so that linting would not demand edits to a transcript: it is the right call for fidelity, and it also removes the one tool that would have shown the file is live code. Tier 3 rather than 2 because it needs a specific action (interpreter started in that directory) rather than arriving on its own. **Exit, applied:** a retained artefact may not carry a name the runtime treats as special; prefix it (`probe_`), and record the original name beside it, since for a transcript the name is part of the finding. Cross-refs: **C-171** (no identity separates current from left-over), **#648**. |

---

### D-01 — Intentional config duplication vs. DRY principle

| Field | Value |
|---|---|
| **Trigger** | Partition boundary update requires editing 73 files atomically |
| **Source** | expert-code-review (Martin vs. Ousterhout/Hickey) |
| **Status** | **Resolved (closed during review-rr, 2026-07-31)** — the resolution was not just chosen, it was **built** |
| **Notes** | Martin (Clean Code) considers 73 identical files a DRY violation creating coordination nightmares. Ousterhout (Complexity) and Hickey (Simplicity) support the duplication because it eliminates shared-state reasoning and keeps each model self-contained. Resolution: the duplication is load-bearing; build a migration tool rather than centralizing. Related to C-01. **Closed 2026-07-31:** the disagreement is settled in practice and in code. `meta/partitions.json` is the single source of truth and `tools/partitions/bump.py` is the migration tool the resolution called for — 37 unit tests, 3 falsification rounds, invariant validation, atomic writes, a JSONL lockfile — and C-01 is correspondingly de-tiered (1 → 3) in this same pass. The Ousterhout/Hickey position is now a **standing repo convention**: model config files stay self-contained and are NOT centralized; centralizing `config_partitions.py` is explicitly out of bounds. Nothing remains in tension — keeping this Open implied a live decision that was made and executed 3+ months ago. Reopen only if someone proposes centralizing configs again, in which case this entry is the prior. |

---

### D-02 — Hardcoded algorithm-to-package mapping vs. factory pattern

| Field | Value |
|---|---|
| **Trigger** | A new algorithm is added and the test mapping must be manually updated |
| **Source** | expert-code-review (GoF vs. Beck/Hickey) |
| **Status** | Open (deferred — closure trigger added 2026-07-31) |
| **Notes** | Gang of Four would prefer a factory in `views_pipeline_core` that maps algorithm→manager, eliminating the need for `ALGORITHM_TO_PACKAGE` in `test_algorithm_coherence.py`. Beck accepts the mapping as pragmatic (test failure = correct signal). Hickey prefers data (dict) over abstraction (factory). Resolution: correct for this repo's scope; factory is a cross-repo decision for `views_pipeline_core`. **Closure trigger added during review-rr (2026-07-31)** — the resolution deferred to another repo without saying when to look again, which is how a deferral becomes a permanent Open. **Revisit when either:** (a) `views_pipeline_core` ships an algorithm→manager registry or factory (at which point delete `ALGORITHM_TO_PACKAGE` and close as GoF-resolved), **or** (b) `ALGORITHM_TO_PACKAGE` needs a fourth manual edit in one release cycle (at which point the maintenance cost has outgrown Beck's pragmatism and views-models should ask pipeline-core for the registry). Until one of those fires, the hardcoded dict stands and this entry needs no attention. |

---

### D-03 — `config_queryset.py` dependency exception: essential or architectural violation

| Field | Value |
|---|---|
| **Trigger** | Decision to refactor config loading or extend test coverage to querysets |
| **Source** | expert-code-review (Martin vs. Kleppmann vs. Ousterhout) |
| **Status** | Open (resolution is theoretical — closure tied to C-02 on 2026-07-31) |
| **Notes** | Martin considers `config_queryset.py`'s external dependencies an architectural boundary violation — configs should be pure. Kleppmann notes it's where data correctness is defined and can't be simplified away. Ousterhout acknowledges the mental tax but accepts it as irreducible complexity. Resolution: the dependency is essential (querysets require the `viewser` DSL). The gap is in testing — AST-based validation of column structure could create a testable seam without requiring external packages. Related to C-02, C-06. **Honest status, review-rr (2026-07-31):** the *dependency* half is genuinely settled (Kleppmann/Ousterhout prevailed; C-06 is `Accepted` under ADR-002). The *testing* half is *theoretical* — the AST seam this resolution names has been the proposed exit for **C-02 since April 2026** and nobody has started it; C-02 is still Open and is a member of **Cluster A**. **Closure now tied to C-02:** this entry closes when C-02's minimum-viable test lands (`generate()` exists, returns the declared type, datafactory descriptors carry `source`/`zarr_url`/`features`) — or, if that seam is judged not worth building, it closes as *"testing gap accepted, no AST seam"*, which is a legitimate answer but must be stated rather than left implied by inaction. |

---

### D-04 — Static analysis tests vs. behavioral execution tests

| Field | Value |
|---|---|
| **Trigger** | A model passes all pytest structural tests but fails at runtime |
| **Source** | test-review (Beck vs. Nygard) |
| **Status** | **Subsumed by C-106 (review-rr, 2026-07-31)** |
| **Notes** | The test suite is almost entirely static analysis (AST parsing, importlib loading, regex extraction). Beck notes this gives exceptional speed (1.41s for 2374 tests) and clean behavioral contracts. Nygard counters that the gap between "structure is correct" and "system works" is wide and uncovered — no `main.py` is ever executed, no training pipeline is ever triggered. The suite validates the blueprint but never builds the house. Related to C-03, C-15. **Subsumed 2026-07-31:** C-106 ("STRATEGIC ROOT: the test architecture verifies declarations exhaustively but runtime behavior nowhere in CI — the config-vs-behavior gap") states the identical finding, carries the same Beck/Nygard framing, names the cluster of ~10 entries that are its instances, **and has a partially-built exit** (`tests/test_runtime_smoke.py` + `runtime_smoke.yml`, PR #272 — 21 baseline models executed end-to-end at PR time). This is a resolved tension, not a live disagreement: Nygard's position prevailed and work started on it. Tracking it in two places split the evidence. Following the C-108 → Appwrite Seam Contract precedent, ownership moves to C-106; this entry is retained as the historical record of where the finding was first named. Do not fix from here — fix from C-106, **Cluster A**. |

---

### D-05 — Is the 131-files-to-11-environments mismatch a defect, or a resource necessity?

| Field | Value |
|---|---|
| **Trigger** | A proposal to give each model its own environment, or to reduce the number of `requirements.txt` |
| **Source** | expert-code-review (2026-08-02; Ousterhout/Kleppmann vs. Nygard) |
| **Status** | Open |
| **Notes** | **Ousterhout and Kleppmann:** the root defect. 131 declaration points imply 131 configuration points; there are 11, so the interface lies, provenance is unrecoverable, and a reader must know three facts that are not in the file they are reading. **Nygard dissents on cost:** `envs/views-baseline` is 9.0G and only 3 of 11 environments exist on the maintainer's laptop; 131 environments would be 100-200G per machine, on laptops, for a team with no ops engineer. Sharing is the only thing that fits the hardware. **Provisional resolution:** both hold, and the fix is neither more environments nor fewer files — the environment count stays at 11 and what changes is that its contents become a declared, committed artifact (**C-116**, **C-117**). Recorded rather than settled because the resolution has not been built. |

---

### D-06 — Does a repo-wide hygiene test that starts by accepting today's exceptions have value, or is it governance theatre?

| Field | Value |
|---|---|
| **Trigger** | Writing `tests/test_requirements_hygiene.py`, or any repo-wide invariant test over the 131 `requirements.txt` |
| **Source** | expert-code-review (2026-08-02; Hickey vs. Beck/Feathers) |
| **Status** | Open |
| **Notes** | **Hickey:** an allowlist of accepted exceptions is a place to hide, and its length is the metric — a test that begins by blessing the mess has inverted its own purpose. **Beck and Feathers:** the objection is about *size*, and size is a choice of ordering. Fix the one unparseable specifier (#316), then the three missing trailing newlines, then the 27 unbounded ceilings (**C-118**) — each rule is red for a real reason, gets fixed, and goes green with no baseline at all. Only the divergent-spec rule needs a recorded exception, and after that sequence it holds one entry: the r2darts split (**C-115**), with its reason. **Martin adds** the deciding criterion: an exception carrying a written reason is documentation; an exception without one is theatre. **Provisional resolution:** build in Beck's order and the disagreement does not arise; revisit if the exception list ever exceeds two entries. |

---

### D-07 — Is the delivery defect structural, or observational?

| Field | Value |
|---|---|
| **Trigger** | Choosing between a declared composition (structure) and a freshness assertion (observation) as the next change |
| **Source** | expert-code-review (2026-08-04; Nygard/Beck vs Martin/Kleppmann/Ousterhout) |
| **Status** | Open |
| **Notes** | **Nygard and Beck:** the incident that cost 145 days was not a wrong order — it was a delivery step that never ran, and which would have republished March data without objecting. A correctly ordered manifest would not have delivered anything either. So the assertion (**C-121**) addresses the failure that happened and the structure (**C-122**) does not. **Martin, Kleppmann and Ousterhout:** an unrepresented causal dependency in a partner-facing pipeline is a defect whether or not it has fired, and leaving it invites the next silent failure. **Provisional resolution:** both, in that order — the assertion now, the structure behind C-122's named trigger. Recorded because the ordering is the actual decision, and it is easy to reverse it in the name of tidiness. |

---

### D-08 — Does WET-before-DRY forbid a declared composition?

| Field | Value |
|---|---|
| **Trigger** | Proposing a composition manifest, a dependency resolver, or moving composition into views-pipeline-core |
| **Source** | expert-code-review (2026-08-04; Hickey vs Martin/Beck) |
| **Status** | Open |
| **Notes** | **Hickey:** views-models has exactly **one** composition (`monthly_run.sh`). Abstracting at n=1 is precisely the wrong-abstraction risk the rule exists to prevent, and a manifest that acquires conditionals has become a program. **Martin and Beck:** the second instance already exists one repo away — `crafd/` was cloned from `unfao/` per `docs/CLONING.md`, and views-postprocessing #211 is the recorded scar of that clone (*"every partner-scoped guard was scoped to ONE partner"*). **Provisional resolution:** the trigger has fired for views-postprocessing, **not** for views-models. Defer, behind C-122's named trigger. When it fires, copy views-postprocessing's remedy — a declared list asserted against the filesystem — not a framework. Unanimous against a dependency resolver and against moving composition into pipeline-core. |

---

### D-09 — Should the `REQUIRE` block be mandatory, or is it ceremony?

| Field | Value |
|---|---|
| **Trigger** | Writing a delivery file that has nothing to assert |
| **Source** | expert-code-review (2026-08-04; Martin/Ousterhout vs Nygard/Kleppmann) |
| **Status** | Open |
| **Notes** | **Martin and Ousterhout:** make the block optional. A `REQUIRE` holding one line will read as boilerplate, and the first person who deletes an empty one teaches everyone else to delete theirs. A block that is always present stops carrying information. **Nygard and Kleppmann:** make it mandatory — `max_age` and `reconciled` are exactly the assertions whose absence caused real failures, and optional safety is not safety. **Provisional resolution (written into ADR-017 §5):** the *block* is optional; specific *rules* are conditional on the delivery's shape — `reconciled` is required with two or more sources, `max_age` is required when `intent = live()`. Requirement follows from what the delivery is, not from ceremony. Revisit if a delivery file appears with an empty `REQUIRE`. |

---
