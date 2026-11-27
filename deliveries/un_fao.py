"""The UN FAO delivery.

**Written as a characterisation, not a change** (ADR-017 §11 Phase 1). Every value here
describes what `postprocessors/un_fao/` already does today. It is no longer inert: since #347/#348
and ADR-021 the FAO config derives its source, arming and coverage from here.
`tests/test_deliveries_characterisation.py` pins that agreement against the *committed*
FAO config, which is what makes views-models#347 and #348 safe to attempt.

Two values are declared here that the committed config does not carry, and they are
called out rather than smuggled:

- **`coverage`** — `postprocessors/un_fao/configs/config_meta.py` carries
  `"region": "land_gaul"` in a working tree but not in git (register C-110, the same
  defect as `wire_upload_enabled`). Declared because it is what runs; excluded from the
  parity test because the repository cannot prove it.
- **`max_age`** — no freshness bound exists anywhere today. That absence is why a
  partner received nothing for 145 days while a complete forecast sat on the shelf
  (#320, register C-121). Two months is the smallest bound that tolerates one late
  monthly run without tolerating a silent quarter.

**On `since`.** It is the date this delivery was *declared here*, not the date it became
live — the delivery predates this file and its true start is recorded nowhere. So any
silence computed from this value is a **lower bound**, not the full gap. The evidence
that it was working earlier is in `docs/forecast_delivery_map.md`: `unfao_bucket`'s
newest forecast dataset is 2026-03-10. Nothing here should be read as claiming the
delivery started today.
"""

from datetime import date

from deliveries.vocabulary import (
    Delivery, Require, pgm, live, monthly, prod, months,
)

DELIVERY = Delivery(               # DECIDES  — change a line, something different happens
    send      = [pgm("rusty_bucket")],
    frequency = monthly,
    tier      = prod,
    intent    = live(since=date(2026, 8, 4)),
)

REQUIRE = Require(                 # REFUSES  — change a line, a different set is rejected
    targets    = ("lr_ged_sb", "lr_ged_ns", "lr_ged_os"),
    coverage   = "land_gaul",
    max_age    = months(2),
)
