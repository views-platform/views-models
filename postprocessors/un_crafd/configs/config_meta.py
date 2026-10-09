"""Meta configuration for the CRAF'd postprocessor.

**Three of these keys are derived, not typed.** `ensemble`, `region` and
`wire_upload_enabled` are read from `deliveries/un_crafd.py`, which is the single place
this delivery is decided (ADR-019 source, ADR-021 coverage, #348 arming).

**To change what CRAF'd receives, or to arm the upload, edit `deliveries/un_crafd.py`.
Not this file.** There is deliberately no fallback: a silent default would ship the wrong
forecast, or ship one that should not have shipped, to an external partner and say
nothing. Failing loudly is cheaper than any of that (ADR-003).

The keys survive as keys because `views_postprocessing` reads them one repository away —
`crafd/managers/crafd.py` takes `configs["ensemble"]` at `:240` (a subscript: absence is a
`KeyError`, not a default), `configs.get("region")` at `:236`, `:312` and `:419`, and
`configs.get("wire_upload_enabled", product.UPLOAD_ENABLED)` at `:354`. So the decision
moved and the interface did not.

**The upload is disarmed today**, because `deliveries/un_crafd.py` declares
`intent = paused(...)`. Arming is views-crafdapi's D5 (#45), after their D4 dry run. With
it disarmed the manager never constructs a partner store at all, so the run stages
artifacts locally and makes zero store calls — which is what makes a first run safe.
"""

import sys
from pathlib import Path

# The delivery declaration lives at the repository root. run.sh cannot set PYTHONPATH for
# us — the same bootstrap the reconciliation layer uses in each ensemble's main.py.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deliveries.status import (  # noqa: E402  (after the path bootstrap)
    declared_coverage,
    declared_source,
    upload_armed,
)

CONSUMER = "un_crafd"


def get_meta_config():
    """
    Meta data for the postprocessor (algorithm, name, targets, level).

    Returns:
    - meta_config (dict): the postprocessor meta configuration.
    """

    meta_config = {
        "name": CONSUMER,
        "algorithm": "Postprocessor",
        "targets": ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"],
        "level": "pgm",
        # Derived, never typed. Edit deliveries/un_crafd.py to change them.
        "ensemble": declared_source(CONSUMER),
        "wire_upload_enabled": upload_armed(CONSUMER),
        "region": declared_coverage(CONSUMER),
        # The ADR-013 wire dialect, not the legacy parquet. views-crafdapi reads only the
        # wire; `contract/launch_config.py:56` refuses the run outright if this is falsy,
        # so it is a declaration rather than a preference.
        "wire_contract": True,
    }

    # Belt-and-braces, not a reconciliation (ADR-021). Derived values cannot disagree, so
    # this can only fire if someone reintroduces a literal. One line, and what it guards
    # is the region written into the provenance record shipped to an external partner.
    assert meta_config["region"] == declared_coverage(CONSUMER), (
        f"config_meta emits region={meta_config['region']!r} but "
        f"deliveries/{CONSUMER}.py declares {declared_coverage(CONSUMER)!r}.\n"
        f"  Do not fix this by editing the literal — there should not be one.\n"
        f"  See docs/ADRs/021_coverage_is_declared_once.md."
    )
    return meta_config
