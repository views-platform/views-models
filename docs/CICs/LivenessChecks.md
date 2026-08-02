# Class Intent Contract: `tools.liveness` surface-check layer

**Status:** Active
**Owner:** Project maintainers
**Last reviewed:** 2026-07-21
**Related ADRs:** ADR-006 (Intent Contracts), ADR-005 (the `Live` test category), ADR-003 (fail-loud), ADR-017 (the observability instrument behind derived `deployed`)

---

## 1. Purpose

> Answer, with raw facts, whether the VIEWS forecasting system is alive on every input
> and output surface. One module per external surface; one command
> (`python -m tools.liveness`) that runs them all and exits with the worst verdict.

> Located in: `tools/liveness/`

Built as epic #238 after the 2026-07-19 episode in which nobody — human or AI — could
check whether the forecasts were live, and un-encoded conventions produced false alarms.
Each surface reports RAW FACTS (a URL hit, a value found, a date derived) plus a verdict
string — never narration.

## 2. Non-Goals (Explicit Exclusions)

- It does **not** narrate or interpret ("the pipeline looks healthy") — it emits facts and a verdict.
- It does **not** fix, alert, retry, or page — it observes and classifies, nothing more.
- It does **not** write files, mutate any store, or run at import time (zero import-time side effects, C-93).
- It does **not** add dependencies (stdlib `urllib`, lazy-imported; optional SDKs skip truthfully).
- It is **not** a monitoring daemon — it is single-shot: run it, read the block, get an exit code.

## 3. Responsibilities and Guarantees

**The shared per-surface contract** (each surface module honours all three):

- `<Surface>Check` — a class whose constructor is `__init__(self, fetch=None)`, the **injected
  fetch/dependency seam** (DIP; mirrors `reconciliation/viewser_country_mapping_provider.py`),
  defaulting to a stdlib lazy-`urllib` fetch. No import-time work.
- `<Surface>Check.run(...) -> CheckReport` — a `@dataclass(frozen=True)` of raw facts carrying a
  `verdict: str`. Deterministic-test hooks (e.g. `now_month_id`) are injectable.
- `main(fetch=None, ...) -> int` — **classifies before printing**: it calls `exit_code_for(verdict)`
  *first* (so an unregistered verdict raises loud before a half-block prints — C-101/P7), then prints
  one fact per line, then returns the exit code. Guarded by `if __name__ == "__main__"`.

**The six surfaces** (epic #238; verdict catalogue in `tools/liveness/README.md`):

- `old_api.OldApiCheck` — the public API `api.viewsforecasting.org`: newest fatalities run fresh, and serving rows at **both** `cm` and `pgm` levels.
- `datafactory_input.*` — the datafactory zarr input store: observed coverage vs the requirement **derived from `meta/partitions.json`** (re-arms on every partition bump — automates the C-96 tripwire).
- `appwrite_store.*` — the internal Appwrite `production_forecasts` shelf: is anything landing, and does the real metadata collection exist (server-side `orderDesc($createdAt)`, never the 25-per-page default — the #241/#242 false-idle fix).
- `unfao_delivery.*` — the FAO partner `unfao_bucket`: per-stream freshness of `forecast_dataset_*` and `historical_dataset_*`, judged independently.
- `wandb_execution.*` — did the team actually compute this cycle (execution recency)?
- `vpn_store.*` — the `gjoll` store behind the VPN: truthful `VPN_REQUIRED` when off-network.

**The shared report contract** — `report.py`:

- `EXIT_CODE_BY_VERDICT` — the verdict enum, as dict keys → uniform exit code (§6).
- `exit_code_for(verdict) -> int` — the classifier; **raises `KeyError` on an unregistered verdict** (fail-loud, ADR-003).
- `render_facts(facts)` / `one_line(value)` — the one-`key: value`-per-line renderer; embedded newlines collapse to `\n`; `None` facts are omitted.
- `worst_exit(codes) -> int` — the aggregate exit code is the worst of the parts (empty → 0).

**The aggregate runner** — `__main__.run_all()`:

- Runs every surface in `SURFACES` in sequence, prints each block, returns `worst_exit`.
- **Contains crashes**: a surface that raises is reported as an `UNREACHABLE` fact with exit 2 — one broken surface must never hide the others.

## 4. Inputs and Assumptions

- An injected `fetch` callable (real network by default; a stub in tests).
- Optional environment: Appwrite credentials, the VPN, and optional SDK packages (`datafactory_query`, appwrite). Absence is a **fact about the environment**, not a failure (§6).
- An injectable clock (`now_month_id`) for deterministic freshness math.
- Month arithmetic is **reused** from `tools.partitions.domain` (`date_to_month_id`, `month_id_to_date`) — not reimplemented.
- The API run-naming convention (`fatalities{gen}_{yyyy}_{mm}_t{seq}`, keyed on **data-cutoff** month) is encoded once in `old_api`, cited to the `views_api` wiki — misreading it as execution-month once produced a false "stalled" alarm.

## 5. Outputs and Side Effects

- **stdout**: one `key: value` fact per line per surface; the aggregate appends `worst_exit: N`.
- **process exit code**: `0` / `1` / `2` per §6.
- **No file writes, no store mutation, no import-time side effects** (C-93). The only side effect is the network read the injected `fetch` performs.

## 6. Failure Modes and Loudness

| Condition | Behaviour |
|---|---|
| A verdict not in `EXIT_CODE_BY_VERDICT` | `exit_code_for` raises `KeyError` — loud — and because `main` classifies **before** printing, no contradictory half-block is emitted (C-101/P7). |
| A surface's network/parse fails | Contained inside that surface as a `verdict=UNREACHABLE` fact (exit 2), never an uncaught crash. |
| A surface `main` itself crashes | The aggregate runner catches it, prints an `UNREACHABLE` fact, assigns exit 2 — the other surfaces still run. |
| Missing credentials / package / VPN | A **truthful skip** (C-75): `SKIP_NO_CREDENTIALS` / `SKIP_NO_PACKAGE` / `VPN_REQUIRED` → exit **0**. Not-observed is not the same as not-live. |
| **Partially** configured credentials (#298) | `CREDENTIALS_INCOMPLETE` → exit **1**, naming the missing variables. Deliberately *not* a truthful skip: "nothing is configured" is an honest absence of observation, "configured, but half of it" is a fault a human must fix. Exit 1 (attention), not 2 — the world is reachable; our configuration is not right. Collapsing the two is what allowed the Appwrite surfaces to fall back to another repository's `.env` unnoticed. |
| Credentials resolvable only from **another repository's** `.env` | Not resolvable. The Appwrite surfaces read process env, then **this repository's own** `.env` (`REPO_ROOT/.env`, either `KEY=` or `export KEY=` style) — and nothing else. Observing under a foreign identity answers a different question than the one the verdict reports, and an exit code carries no caveat. Pinned by `tests/test_liveness_appwrite_store.py::test_resolve_credentials_does_NOT_read_another_repos_env`. |
| **A rejected Appwrite key** (expired, revoked, wrong) | `UNREACHABLE` → exit **2**. Not negotiable, and not free: Appwrite answers the **file-listing** endpoint with HTTP 200 and `total: 0` for a rejected key — measured 2026-08-02 against Appwrite 1.9.5 (real key → 200/total=461; garbage key → 200/total=0; empty key → 200/total=0). Listing files was the only call the Appwrite surfaces made, so a dead credential was indistinguishable from an empty bucket, and they reported `STORE_IDLE` / `DELIVERY_STALLED` — exit 1, "attention" — while nothing was authenticated. Both keys expire around 2026-11-30 and the write path reports that expiry as success, so these surfaces *are* the detector; rendering that failure as mild staleness defeated their purpose. Every other endpoint returns 401, so `assert_bucket_reachable` **GETs the bucket before any listing is interpreted** — proving key acceptance (401) and coordinate resolution (404) in one call. Pinned by `test_rejected_key_is_unreachable_not_idle` and `test_key_is_verified_before_any_listing_is_believed` in both Appwrite suites. |
| Reachable but stale/idle/not-serving | Verdict maps to exit **1** (attention). |

## 7. Boundaries and Interactions

- **Depends on** `tools.partitions.domain` (month math) and, at run time, the real surfaces (API, datafactory, Appwrite, wandb, VPN store) via the injected `fetch`.
- **DIP seam** mirrors `reconciliation/` — the fetch is the port; the default stdlib fetch is the concrete; tests inject a stub. No new cross-repo coupling, no new dependency.
- **Consumed by** ADR-017's observability decision: a delivery surface counts as `deployed` only when its liveness surface is green (ADR-017 §7). The `live` test category (ADR-005) is the CI-side companion — one skip-truthful probe per surface.
- Import-time purity (C-93) keeps `python -m tools.liveness.<surface>` cheap and side-effect-free.

## 8. Examples of Correct Usage

```bash
# Whole dashboard — exit = worst verdict across all surfaces:
conda run -n views_pipeline python -m tools.liveness

# A single surface:
conda run -n views_pipeline python -m tools.liveness.old_api
```

```python
# Deterministic test: inject the fetch and the clock, assert on facts.
report = OldApiCheck(fetch=fake_fetch).run(now_month_id=557)
assert report.verdict == "LIVE_FRESH"
assert exit_code_for(report.verdict) == 0
```

## 9. Examples of Incorrect Usage

```python
# Wrong: emitting a verdict that is not registered in EXIT_CODE_BY_VERDICT.
# exit_code_for("LOOKS_OK") raises KeyError — add the verdict to the map (and here).

# Wrong: print(render(report)) BEFORE classifying — a bad verdict would print a
# block the runner then contradicts. main() must call exit_code_for() first (C-101/P7).

# Wrong: doing network work at import time, or writing a file — breaks C-93 / §2/§5.
```

## 10. Test Alignment

- Per-surface: `tests/test_liveness_old_api.py`, `..._datafactory_input.py`, `..._appwrite_store.py`, `..._unfao_delivery.py`, `..._wandb_execution.py`, `..._vpn_store.py`.
- Runner: `tests/test_liveness_runner.py` (crash-containment + `worst_exit`).
- Adversarial: `tests/test_liveness_falsifications.py` (the falsify-audit fixes, C-101/C-102).
- Taxonomy: `tests/test_liveness_taxonomy.py` (the ADR-005 `live` marker, C-103).
- 130 tests total across the suite.

## 11. Evolution Notes

- Epic #238 (S1 `old_api` … S7 runner/DRY extraction of `report.py` … S8 README); pgm-probe added the second data level to `old_api` (#260).
- Register lineage: C-100 (Mitigated, README), C-101/C-102 (falsify audit), C-103 (taxonomy, Resolved), **C-107** (this contract gap — closed by this CIC).
- ADR-005 amendment 2026-07-19 added the `Live` category so the skip-truthful probes are a first-class axis, not mislabeled `red`.

## 12. Known Deviations

- `report.py` was extracted **WET-before-DRY** (S7/#245): the shared renderer and classifier were pulled out only after six surfaces demonstrably duplicated them — deliberate, per epic #238.
- The per-surface **verdict catalogue** (every verdict and its freshness budget) lives in `tools/liveness/README.md` as the human-readable companion and is **referenced, not duplicated**, here — this contract governs the *shared* shape; the README enumerates the *specifics*.

---

## End of Contract

This document defines the **intended meaning** of the `tools.liveness` surface-check layer.

Changes to behavior that violate this intent are bugs.
Changes to intent must update this contract.
