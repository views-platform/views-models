"""The CRAF'd delivery.

The second consumer. `postprocessors/un_crafd/` derives everything it needs from this
file — source, coverage and arming are read from here, never typed there (ADR-019,
ADR-021). Changing a line here changes what the launcher does; there is nothing to keep
in step by hand.

**The source is `chrome_box`, and it is a reconciled forecast.** `chrome_box` is the pgm
grid ensemble; it declares `reconciliation: "pgm_cm"` and `reconcile_with:
"chrome_crate"` in its own `config_meta.py`, and reconciles its pooled draws against
`chrome_crate`'s cm country forecast draw by draw *during its own run*. What reaches the
shelf, and therefore CRAF'd, is one pgm wire that has already been reconciled. That is
what `reconciled=True` below asserts.

**Why `send` names one source and not both — this was tried.** ADR-017 §3 and ADR-019 §4
do contemplate sending two: a consumer may want the grid forecast *and* the country
forecast it was reconciled against, and ADR-019 §3 then requires `reconciled=True`
(`coherence.py:390` raises otherwise). Coherence accepts `[pgm("chrome_box"),
cm("chrome_crate")]` happily. `declared_source` does not, and the limit is one
repository away: `views_postprocessing` reads `configs["ensemble"]` as a single string
(`crafd/managers/crafd.py:240`, a subscript, so absence is a `KeyError`) and has no key
for a list. Naming both makes `postprocessors/un_crafd/configs/config_meta.py` raise at
import, so the launcher cannot load at all.

So two sources is not refused by any rule here — it is unimplemented downstream, and
sending the cm stream to CRAF'd would need views-postprocessing to carry two sources and
views-crafdapi to accept a second level. Neither is asked for: the ADR-013 wire CRAF'd
reads is 108 pgm shards (3 targets × 36 months), and `chrome_crate`'s own output is an
input to the reconciliation, not a thing CRAF'd has asked to receive. If it is ever
wanted, that is a two-repository change and Simon's call.

**This closes the modelling gap #422 named.** That issue recorded that reconciliation
existed for `lr_ged_sb` only — no cm forecast for `ns` or `os` existed anywhere in the
repository, so a three-target reconciled delivery was not merely unexpressed but
impossible. `chrome_crate` is a cm ensemble carrying all three, which is #422's option 1.

**Why it is no longer `rusty_bucket`.** It was, and `intent` was `live` — so a hand-run
of this postprocessor would have delivered FAO's ensemble to CRAF'd. Simon's ruling of
2026-10-22 is that `rusty_bucket` is FAO's and must never go to CRAF'd as a stopgap.
Nothing had fired only because `monthly_run.sh:179` runs `un_fao` and not this one;
that is luck, not a safeguard.

**Declared `paused`, and it must stay paused until ONE thing lands plus one run.**
Both code blockers are now closed; what remains is a partner's gate and the fact that
no reconciled wire has been produced yet.

  - ~~views-pipeline-core#562~~ — **fixed and PUBLISHED in 3.3.5, 2026-10-29T20:41:59Z.**
    `load_cm_frame` looked for a `.parquet` file where a PredictionFrame ensemble writes
    a directory, so the cm partner could not be loaded. `requirements.txt` here now
    declares `views-pipeline-core>=3.3.5,<4.0.0` — raised in the same change that
    rewrote this paragraph, because a floor of `>=3.2.0` is satisfied by 3.3.4 and
    `run.sh`'s dry-run gate would have upgraded nothing.
    ⚠️ An earlier version of this paragraph said "Fixed but NOT RELEASED, as of
    2026-10-29", which was true when written and false eight hours later. Re-arming
    needs 3.3.5 *published*, not merely written — the distinction that cost five months
    on #320 — and it now is; verified against PyPI's JSON and simple indexes, `pip index
    versions`, a resolving `pip install --dry-run`, and the cut release.
    Their verification turned up a **second** live symptom beyond the one filed: a PF
    cm *model* partner (as opposed to an ensemble) writes per-target parquets, and the
    unfiltered `paths[0]` read the alphabetically-last target, failing with a missing
    `pred_<target>` column. So a cm partner of either shape was broken, differently.
  - views-crafdapi#180 — whether a 1,024-draw month serves inside their container is
    their own open gate. At 1,024 draws a month is ~0.80 GB raw, inside their 4 GiB cap
    by design, but unverified in the box they run.

  - ~~views-models#647~~ — **fixed 2026-10-29.** The GAUL mapping stopped at the
    datafactory store horizon (month 564) while the forecast window ran to 622, so
    reconciliation died on month 565, step 4 of 36. The mapping is now keyed on the
    window the caller asked for; see
    `reconciliation/datafactory_country_mapping_provider.py`'s
    `_broadcast_over_the_requested_window`.

Both defects were verified end to end on a pod on 2026-10-28 by patching past them:
all three targets reconciled, and the output checked to be genuinely reconciled rather
than merely written (grid draws sum to the cm country forecast, worst relative error
1.2e-07, holding at month 597). So the chain is proven and the blockers are specific.

views-postprocessing#320 is **verified and needs nothing**: the wire target names
normalise. pipeline-core's `INTERNAL_TO_WIRE_TARGET` maps these ensembles' internal
`lr_*_best` to the canonical `lr_ged_*` before anything is written, which is what
`REQUIRE.targets` below names and what the consumer expects.

**On `since` / the pause date.** `live(since=2026-08-14)` was the rusty_bucket arming.
It is deliberately not carried over: this is a different delivery of a different
ensemble, and CRAF'd has never received either.

**Coherence.** `check()` warns on this file: `chrome_box` is `candidate` and a `prod`
consumer wants `graduate` (ADR-017 §5). Same day-one violation the FAO edge carries,
recorded at ADR-017 §11 — not new, and not a reason to graduate anything hastily.
"""
from datetime import date

from deliveries.vocabulary import (  # noqa: F401  (`live` — kept for the re-arm)
    Delivery, Require, pgm, live, paused, monthly, prod, months,
)

# `live` is imported and unused, deliberately — the mirror of what this comment used to
# say about `paused`. Re-arming should be a one-word edit to `intent`, not an edit to
# `intent` AND an import line, and the moment you reach for it is the moment you least
# want a NameError. Same reasoning as un_fao.py.

DELIVERY = Delivery(               # DECIDES  — change a line, something different happens
    send      = [pgm("chrome_box")],
    frequency = monthly,
    tier      = prod,
    # DISARMED 2026-10-28. Was `live(since=2026-08-14)` pointing at `rusty_bucket`,
    # which is FAO's — see the docstring. Re-arm only when all three conditions in the
    # docstring are met, and then with a `since` date of that day, not this one.
    intent    = paused(
        reason=(
            "no reconciled wire has been produced yet, and views-crafdapi#180 "
            "is their own open serving gate at 1,024 draws. Both code blockers are "
            "closed: views-pipeline-core#562 shipped in 3.3.5 (published "
            "2026-10-29) and views-models#647 is fixed."
        ),
        since=date(2026, 10, 28),
    ),
)

REQUIRE = Require(                 # REFUSES  — change a line, a different set is rejected
    # A claim about what chrome_box puts on the shelf, not a formality: it reconciles
    # pgm_cm against chrome_crate draw by draw, verified on a pod 2026-10-28 (worst
    # relative error 1.2e-07). With one source `coherence.py` does not check this —
    # `_check_reconciliation` returns early below two sources — so the guard that keeps
    # it honest is in tests/test_delivery_map_truth.py.
    reconciled = True,
    targets    = ("lr_ged_sb", "lr_ged_ns", "lr_ged_os"),
    coverage   = "land_gaul",
    max_age    = months(2),
)
