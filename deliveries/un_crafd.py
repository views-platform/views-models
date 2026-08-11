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

from deliveries.vocabulary import (
    Delivery, Require, pgm, paused, monthly, prod, months,
)

DELIVERY = Delivery(               # DECIDES  — change a line, something different happens
    send      = [pgm("rusty_bucket")],
    frequency = monthly,
    tier      = prod,
    intent    = paused(
        "crafdapi's first delivery has not been executed; arming is views-crafdapi D5 (#45), "
        "after the D4 dry run",
        since=date(2026, 8, 11),
    ),
)

REQUIRE = Require(                 # REFUSES  — change a line, a different set is rejected
    targets    = ("lr_ged_sb", "lr_ged_ns", "lr_ged_os"),
    coverage   = "land_gaul",
    max_age    = months(2),
)
