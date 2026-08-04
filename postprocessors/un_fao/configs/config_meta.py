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
"""

import sys
from pathlib import Path

# The delivery declaration lives at the repository root. run.sh is immutable, so
# PYTHONPATH cannot be set there — the same bootstrap the reconciliation layer uses in
# each ensemble's main.py (see ensembles/skinny_love/main.py:9).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _declared_source() -> str:
    """The single source this consumer is declared to receive.

    Raises rather than guessing. Every failure names the file to open (ADR-020).
    """
    try:
        from deliveries.un_fao import DELIVERY
    except Exception as exc:
        raise RuntimeError(
            f"cannot read the FAO delivery declaration: {exc}.\n"
            f"  Open deliveries/un_fao.py — it declares which forecast goes to the UN.\n"
            f"  This config derives its source from that file and will not guess one."
        ) from exc

    sources = list(DELIVERY.send)
    if len(sources) != 1:
        raise RuntimeError(
            f"deliveries/un_fao.py declares {len(sources)} sources, and this "
            f"postprocessor can carry exactly one.\n"
            f"  Open deliveries/un_fao.py and check `send`.\n"
            f"  Delivering several sources to one consumer needs the reconciliation "
            f"rules in ADR-019 §4, which the postprocessor does not yet implement."
        )
    return sources[0].name


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
        # Derived, never typed. Edit deliveries/un_fao.py to change it.
        "ensemble": _declared_source(),
    }
    return meta_config
