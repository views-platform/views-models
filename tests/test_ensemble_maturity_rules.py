"""ADR-017 §5 R1 and R2, fleet-wide, in both vocabularies.

- R1: no ensemble that is `candidate` or `graduate` contains a `retired` member.
- R2: an ensemble is `graduate` only if every member is `graduate`.

`deliveries/coherence.py` enforces both — but only for the ensembles a delivery names
(C-144). This is the same check over every `ensembles/*/` regardless of delivery, so an
author who promotes an ensemble to `graduate` in `config_maturity.py` while a member is
still `candidate` gets a red test, not a quiet catalog.

History: this file was `test_falsify_deployment_status_convention.py`, two xfail stubs
from the 2026-07 falsification of the `shadow → deployed` convention. Its own xfail
reason said it would flip to a hard gate "when the rule is decided + the violation
resolved". The rule is ADR-017 §5 (2026-07-27); the violation — `white_mustang`
`deployed` over two `shadow` members — was resolved by the rename (#449), which made it
`candidate` under R2. The second stub asserted that pipeline-core's ensemble guard did
not compare against the unsupported literal `'production'`; pipeline-core removed it in
3.2.0 (2026-09-08), so that stub had no subject in any release CI installs.
"""

from pathlib import Path

import deliveries.coherence as coh

REPO_ROOT = Path(__file__).resolve().parent.parent
ENSEMBLES_DIR = REPO_ROOT / "ensembles"


def _ensembles_with_members():
    for ens in sorted(p for p in ENSEMBLES_DIR.iterdir() if (p / "configs").is_dir()):
        modelset = ens / "configs" / "config_modelset.py"
        if not modelset.exists():
            continue
        yield ens.name, coh.source_config(ens.name, "modelset").get("models", [])


def test_r1_no_active_ensemble_contains_a_retired_member():
    violations = [
        f"{ens} ({coh.maturity_of(ens)}) <- {m} (retired)"
        for ens, members in _ensembles_with_members()
        if coh.maturity_of(ens) in ("candidate", "graduate")
        for m in members
        if coh.maturity_of(m) == "retired"
    ]
    assert not violations, f"ADR-017 §5 R1: {violations}"


def test_r2_a_graduate_ensemble_has_only_graduate_members():
    violations = [
        f"{ens} (graduate) <- {m} ({coh.maturity_of(m)})"
        for ens, members in _ensembles_with_members()
        if coh.maturity_of(ens) == "graduate"
        for m in members
        if coh.maturity_of(m) != "graduate"
    ]
    assert not violations, f"ADR-017 §5 R2: {violations}"
