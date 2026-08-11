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


def _declared_region() -> str:
    """The coverage this consumer is declared to receive."""
    from deliveries.un_fao import REQUIRE

    if not REQUIRE.coverage:
        raise RuntimeError(
            "deliveries/un_fao.py declares no coverage, so this postprocessor cannot "
            "confirm which region it would ship.\n"
            "  Open deliveries/un_fao.py and add coverage=... to REQUIRE."
        )
    return REQUIRE.coverage


def _upload_armed() -> bool:
    """Whether this delivery is armed — derived from `intent`, never typed.

    views-postprocessing ADR-013 §11.4 sets ``UPLOAD_ENABLED = False`` and makes the
    launcher key ``wire_upload_enabled`` its only override
    (``unfao/managers/unfao.py:317``). That contract is unchanged; what changed is who
    computes the key. `intent` and a hand-written boolean were the same fact in two
    places, which ADR-019 §8 rejects by name (register C-129).

    **There is no longer a region cross-check here, and that is the point.** Until
    ADR-021 this function compared the delivery's `coverage` against a `REGION` literal
    in `config_queryset.py` — parsed out of that file's source with `ast`, because
    importing it pulls in pipeline-core — and disarmed when they disagreed. Both are
    now derived from `deliveries.status.declared_coverage()`, so they cannot disagree:
    the check was deleted rather than extended to the third copy, because reconciling
    a duplication keeps the duplication.

    The residual assertion in `get_meta_config()` is deliberate belt-and-braces, not a
    reconciliation. It costs one line, and the value it guards is written into the
    provenance record delivered to a UN agency.

    That is what makes a derived arming state safe to commit: a fresh clone cannot
    silently ship the wrong region to a UN agency, and it does not break either.
    """
    from deliveries.un_fao import DELIVERY

    return DELIVERY.intent.state == "live"


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
        "ensemble": _declared_source(),
        "wire_upload_enabled": _upload_armed(),
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
        "region": _declared_region(),
    }
    # Belt-and-braces, not a reconciliation (ADR-021). Derived values cannot disagree,
    # so this can only fire if someone reintroduces a literal. One line, and what it
    # guards is the region written into the provenance record shipped to the UN FAO.
    assert meta_config["region"] == _declared_region(), (
        f"config_meta emits region={meta_config['region']!r} but "
        f"deliveries/un_fao.py declares {_declared_region()!r}.\n"
        f"  Do not fix this by editing the literal — there should not be one.\n"
        f"  See docs/ADRs/021_coverage_is_declared_once.md."
    )
    return meta_config
