"""The CRAF'd delivery.

The second consumer. `postprocessors/un_crafd/` derives everything it needs from this
file — source, coverage and arming are read from here, never typed there (ADR-019,
ADR-021). Changing a line here changes what the launcher does; there is nothing to keep
in step by hand.

**Declared `paused`, deliberately.** views-models#333 asked for
`wire_upload_enabled: False` on the first build. Since #348 that key is derived from
`intent`, so the way to say it is `paused(...)` — which keeps the reason and the date
visible, where a deleted delivery or a hand-written `False` would keep neither.
views-crafdapi's epic sequences the flip to `live` as its own story (their D5, #45)
after a dry run (D4, #44); it is not this file's decision to pre-empt.

**Why the same source as FAO.** `rusty_bucket` feeds both partners. The two deliveries
differ in destination and in nothing else today — same three targets, same coverage,
same monthly cadence. That is a fact about the current platform, not a constraint: the
whole point of one file per consumer is that CRAF'd can diverge without touching FAO.

**On `since`.** The date this delivery was declared, which is also the date the launcher
was built — CRAF'd has never received anything, so unlike `un_fao.py` there is no earlier
true start being approximated here.

**Coherence.** `check()` warns on this file: `rusty_bucket` is `candidate` and a `prod`
consumer wants `graduate` (ADR-017 §5). That is the same day-one violation the FAO edge
carries, recorded at ADR-017 §11 — not a new one, and not a reason to graduate anything
hastily. It is a warning, not a failure, for exactly that reason.
"""

from datetime import date

from deliveries.vocabulary import (  # noqa: F401  (`paused` — see below)
    Delivery, Require, pgm, live, paused, monthly, prod, months,
)

# `paused` is imported and unused, deliberately. Disarming this delivery should be a
# one-word edit to `intent`, not an edit to `intent` AND an import line — and the moment
# you reach for it is the moment you least want a NameError. Same reasoning as un_fao.py.

DELIVERY = Delivery(               # DECIDES  — change a line, something different happens
    send      = [pgm("rusty_bucket")],
    frequency = monthly,
    tier      = prod,
    # ARMED 2026-08-14 — views-crafdapi D5 (#45), on the maintainer's decision.
    #
    # `paused` since 2026-08-11 because crafdapi's first delivery had not been executed
    # and their D4 dry run (#44) had not passed. It has now: D4 closed with all 10
    # preflight gates green and a SERVABLE verdict, against a staged run this launcher
    # produced with the interlock closed — 108 shards, sidecar, manifest, and a 163.9 MB
    # historical artifact, 108/108 checksums, `provenance.ensemble = "rusty_bucket"` on
    # every shard.
    #
    # The four views-models defects that broke their 2026-08-12 attempt are fixed: #386
    # (views-datafactory uninstallable on py3.11 — resolved upstream by 1.12.0), #392 (a
    # failed install did not stop the run), #385 (the pin was not applied), and #391
    # (this launcher was pinned to 3286eab, whose upload check fails open, on a prefix
    # shared with the armed FAO delivery — C-139).
    #
    # A FIFTH, and the most recent: their 2026-08-13 attempt died on
    # views-postprocessing#268 — the store port's `download` chained `.get()` onto an
    # unvalidated result, so a present-and-null `data` raised AttributeError three frames
    # away, naming neither the file_id nor the failed download. Fixed by moving both
    # launchers to `1.1.1` (views-models#403, S2 of #412) — both, because they share one
    # conda prefix, so moving only this one would leave the armed FAO leg on the defect.
    # Verified before merging this: with both deliveries armed on 1.1.1 the pin-safety
    # guards pass 6/6; on 1.1.0 they fail 4/6. That is why #403 had to land first.
    #
    # `paused` stays imported: disarming is one word, and the reason to reach for it is
    # exactly the kind of moment when you do not want to be editing an import list.
    intent    = live(since=date(2026, 8, 14)),
)

REQUIRE = Require(                 # REFUSES  — change a line, a different set is rejected
    targets    = ("lr_ged_sb", "lr_ged_ns", "lr_ged_os"),
    coverage   = "land_gaul",
    max_age    = months(2),
)
