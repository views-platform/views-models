"""Meta configuration for the UN FAO postprocessor.

**What this file decides, and what it no longer decides.**

It used to carry a line of the form `"ensemble": "<a source name>"` — one hand-typed string choosing
which forecast reaches the UN — under a docstring claiming that modifying this file
"will not affect the model". That was false, and it is the specific defect ADR-019 was
written to remove.

The source is now **declared once**, in `deliveries/un_fao.py`, and *derived* here. The
key survives because `views_postprocessing` reads `self.configs["ensemble"]`
(`unfao/managers/unfao.py:140` and `:195`) one repository away; removing it would raise
`KeyError` there. So the decision moved and the interface did not — ADR-017's principle
applied literally: a thing that is never typed cannot lie.

**To change which forecast goes to the UN, edit `deliveries/un_fao.py`. Not this file.**

There is deliberately no fallback. If the declaration is missing or malformed this fails
loudly (ADR-003): a silent default would deliver the *wrong forecast to a UN agency* and
say nothing, which is worse than any error this file could raise.

**The three accessors are `deliveries.status`'s, not this file's (#430).** They were
hand-copied here — a fourth copy of logic that already had a home, in the one file whose
job is to have no copies. The CRAF'd config has called the shared ones since it was
written; this file now matches it.

They had **diverged**, and not only in the exception type. The copies did
`from deliveries.un_fao import DELIVERY`, so they read whatever was in `sys.modules`. The
shared accessors re-execute the declaration **from disk** on every call. In a normal run
these are the same answer; they differ when something has already imported the module.
The values `get_meta_config()` emits are byte-identical either way — checked — but two
tests had been simulating a broken declaration by patching `sys.modules`, which no longer
simulates anything. They now edit or delete the file in a repo copy, which is what they
were always describing.

Two pieces of history those copies carried, kept because nothing else records them:

- **`wire_upload_enabled` used to compare the delivery's coverage against a `REGION`
  literal in `config_queryset.py`** — parsed out with `ast`, because importing that module
  pulls in pipeline-core — and disarmed when the two disagreed. Both are now derived from
  `declared_coverage()`, so they cannot disagree, and the check was **deleted rather than
  extended to a third copy**: reconciling a duplication keeps the duplication (ADR-021).
- **`region` was a literal until 2026-08-11**, and was the one of its three copies that
  nothing checked — the copy the manager actually reads (register C-110, C-133).
"""

import sys
from pathlib import Path

# The delivery declaration lives at the repository root. run.sh is immutable, so
# PYTHONPATH cannot be set there — the same bootstrap the reconciliation layer uses in
# each ensemble's main.py (see ensembles/skinny_love/main.py:9).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deliveries.status import (  # noqa: E402  (after the path bootstrap)
    declared_coverage,
    declared_source,
    upload_armed,
)

CONSUMER = "un_fao"


def get_meta_config():
    """
    Meta data for the postprocessor (algorithm, name, targets, level).

    `ensemble` is **derived** from deliveries/un_fao.py, not declared here — see the
    module docstring.

    Returns:
    - meta_config (dict): the postprocessor meta configuration.
    """

    meta_config = {
        "name": "un_fao",
        "algorithm": "Postprocessor",
        "targets": ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"],
        "level": "pgm",
        # Both derived, never typed. Edit deliveries/un_fao.py to change them.
        "ensemble": declared_source(CONSUMER),
        "wire_upload_enabled": upload_armed(CONSUMER),
        # run-0 contract-leg delivery — views-postprocessing instruction 2026-07-27:
        # read/deliver the ADR-013 wire dialect and un-freeze the upload interlock,
        # with the declared land_gaul region curation applied at the delivery boundary.
        # NOT COMMITTED (register C-110): wire_contract and region are still
        # working-tree only. wire_upload_enabled is no longer here — it is derived
        # from DELIVERY.intent above (#348).
        "wire_contract": True,
        # Derived, never typed (ADR-021). This is the copy the manager actually reads
        # — views-postprocessing unfao/managers/unfao.py:236,312,419 — and the one
        # written into the delivered provenance record (delivery/provenance.py:47).
        # It was a literal until 2026-08-11, and the only one of the three copies of
        # this string that nothing checked (register C-133).
        "region": declared_coverage(CONSUMER),
    }
    # Belt-and-braces, not a reconciliation (ADR-021). Derived values cannot disagree,
    # so this can only fire if someone reintroduces a literal. One line, and what it
    # guards is the region written into the provenance record shipped to the UN FAO.
    assert meta_config["region"] == declared_coverage(CONSUMER), (
        f"config_meta emits region={meta_config['region']!r} but "
        f"deliveries/un_fao.py declares {declared_coverage(CONSUMER)!r}.\n"
        f"  Do not fix this by editing the literal — there should not be one.\n"
        f"  See docs/ADRs/021_coverage_is_declared_once.md."
    )
    return meta_config
