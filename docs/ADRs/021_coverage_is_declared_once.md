# ADR-021: Coverage is declared once, in the delivery

**Status:** Accepted
**Date:** 2026-08-11
**Deciders:** Simon, VIEWS platform team
**Related ADRs:** [ADR-017](017_source_composition_delivery.md) (the three axes), [ADR-019](019_delivery_declaration.md) (the delivery file format; §3 the vocabulary, §8 "two places to state one fact"), [ADR-020](020_errors_must_descend.md) (errors name the next file down)

---

## Context

`land_gaul` — the set of cells the UN FAO receives — was a typed literal in **three**
places:

| # | location | guarded by |
|---|---|---|
| 1 | `deliveries/un_fao.py` — `coverage = "land_gaul"` | `_upload_armed()` |
| 2 | `postprocessors/un_fao/configs/config_queryset.py` — `REGION = "land_gaul"` | `_upload_armed()` |
| 3 | `postprocessors/un_fao/configs/config_meta.py` — `"region": "land_gaul"` | **nothing** |

`config_meta.py::_upload_armed()` compared (1) against (2) and disarmed the upload when
they differed. It never looked at (3) — and (3) is the only one that leaves this
repository. views-postprocessing's `unfao/managers/unfao.py` reads `configs.get("region")`
at lines 236, 312 and 419, and `delivery/provenance.py:47` writes it into the provenance
record delivered to the partner.

So the interlock guarded the two copies the manager never reads, and not the one it does.
Setting (1) and (2) both to `"land"` **armed** the upload while (3) still said
`"land_gaul"`: the run would curate to `land_gaul` and ship a provenance record claiming
`land_gaul` against a declaration saying `land`. No warning, output that looks correct,
external partner. Registered as **C-133**.

Two further facts made this the moment to fix it rather than note it:

- **`_queryset_region()` was a partial Python interpreter.** It could not import
  `config_queryset.py` — that file pulls in pipeline-core — so it walked the file's AST
  for `REGION = <constant>`. It matched only a bare literal; a derived or conditional
  assignment raised `RuntimeError` from a config read. It was knowledge of another file's
  *syntax*, not its interface. Same family as **C-57**.
- **`un_crafd` is the second consumer** (#333, blocked by #373). One postprocessor held
  three copies; two would hold six. This repository's own rule is to extract when a second
  incident shows the shape.

ADR-019 §8 had already rejected the pattern by name — *"two places to state one fact, and
nothing to reconcile them"* — and stated the principle: *"a thing which is never typed
cannot lie."* Coverage was the clause's most literal violation, and the only one that had
been given a *reconciliation* rather than a *derivation*. #348 derived
`wire_upload_enabled` from `DELIVERY.intent`; #360 derived the freshness bound via
`declared_max_age_days()`. Coverage did not get the same treatment.

---

## Decision

### 1. Coverage is declared once, in the delivery.

`deliveries/<consumer>.py`'s `REQUIRE.coverage` is the sole declaration of which cells a
consumer receives. **No `config_*.py` under `postprocessors/` may contain a coverage
literal.**

### 2. Everything else derives it.

`config_queryset.REGION` and `config_meta["region"]` are obtained from
`deliveries.status.declared_coverage(<consumer>)`, which sits beside
`declared_max_age_days()` — the same shape, the same module, the same refusal to default.

The actuals fetch region and the delivered coverage are **one fact**. This is not an
assumption: `config_queryset.py` set `REGION` to the delivery's region precisely because
*"the historical actuals must cover the SAME cells the forecast does"*. Typing it in both
places gave the repository two copies to disagree about, and it did — `africa_me_legacy`
in git against `land_gaul` in a working tree, for seven weeks (**C-110**, closed by #127).

### 3. The producer's extent is a different fact and is not governed here.

`rusty_bucket` forecasts **`land`, 64,818 cells**; the delivery boundary curates that to
**`land_gaul`, 64,742**, by removing 76 sub-Antarctic cells outside FAO GAUL 2024. The
run-0 manifest correctly declares `expected_cell_count: 64818`. That reduction is owned by
`views_postprocessing/delivery/coverage.py`.

**Do not "correct" the 76-cell gap.** Three scopes exist — producer extent, delivered
coverage, actuals fetch — and only the last two are the same fact.

### 4. The cross-check is deleted. One assertion remains.

`_upload_armed()`'s region comparison and `_queryset_region()`'s AST parse are **removed**.
Derived values cannot disagree, so a check for disagreement is dead weight that implies the
state is reachable. Extending it to the third copy would have hardened the duplication
instead of removing it.

A single assertion in `get_meta_config()` — the emitted `region` equals the declaration —
is **kept**, and is deliberately belt-and-braces rather than a reconciliation. It can only
fire if someone reintroduces a literal. It costs one line, and what it guards is delivered
to a UN agency.

---

## Consequences

### Positive

- The failure mode is removed rather than detected. The previous design's best case was a
  correct refusal; this one has no case to refuse.
- A parser of another file's source text is gone, and with it the class of defect where a
  config becomes unreadable because a sibling's assignment stopped being a bare literal.
- `un_crafd` is born derived and types coverage nowhere — #373's decision (b) becomes free
  rather than adding copies four and five.
- Changing coverage is now a one-line edit to the declaration.

### Negative

- **The delivery declaration is now a hard dependency of data fetching.** Before, a
  malformed `deliveries/<consumer>.py` disarmed the upload and the run still produced local
  artifacts; now `config_queryset.generate()` cannot resolve a region and fetching fails.
  This is accepted: a run that cannot say which cells it is fetching should not fetch. Per
  ADR-020, the error names `deliveries/<consumer>.py`.
- `config_queryset.py` gains a `sys.path` bootstrap and an import of `deliveries/`, which
  `config_meta.py` already carried. Both configs now depend on two subsystems.

### Known inconsistency, deliberately not fixed here

The same fact is called **`coverage`** in the declaration and **`region`** in both derived
places, because views-postprocessing reads `configs.get("region")` in six places across
`unfao/managers/unfao.py` and `crafd/managers/crafd.py`. Three names for one fact is part
of how the duplication stayed invisible. Renaming the wire key is a cross-repo change and
is not attempted here; this ADR records the debt rather than hiding it.

### The rule this ADR exists to stop being rediscovered

**A reconciliation is not a derivation.** Cross-checking two copies of a fact leaves the
duplication in place and quietly implies that all copies are covered. Here the check
covered the two that did not matter. If you find yourself adding a comparison between two
places that state the same thing, the fix is to delete one of them.

---

## References

- `deliveries/status.py` — `declared_coverage()`, beside `declared_max_age_days()`
- `postprocessors/un_fao/configs/config_queryset.py`, `config_meta.py` — the derived readers
- `tests/test_intent_arms_the_delivery.py` — the inverted guard: a mismatch must be
  unrepresentable
- Register: **C-133** (the unguarded consumed copy), **C-110** (region uncommitted for
  seven weeks, closed by #127), **C-129** (same fact in two places), **C-57** (a parser
  cannot distinguish code from the text describing it)
- Issues: #127, #333, #373; `views_postprocessing/delivery/coverage.py` for the
  `land` → `land_gaul` curation
