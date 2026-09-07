"""
Maturity Configuration Script — ADR-017 Axis 1.

Maturity answers one question: how finished is this source? It says nothing about where
the forecast goes (that is a delivery, ADR-019) and nothing about what an ensemble
contains (that is `config_modelset.py`).

Values — a closed set of exactly three:
- candidate: in development.
- graduate:  finished, and eligible to write to the production shelf (ADR-017 §4d).
- retired:   dead. No active ensemble may contain a retired member (ADR-017 §5, R1).

`baseline` is NOT a maturity and does not appear here. It is a *role* — a naive yardstick
you score against — carried by `regression_point_baselines` / `regression_sample_baselines`
in `config_meta.py`.

**Both config files exist during the transition and must agree** (ADR-017 §11, the
dual-vocabulary window):

    config_deployment.py                 config_maturity.py
    'deployment_status': 'shadow'    ->  'maturity': 'candidate'
    'deployment_status': 'deployed'  ->  'maturity': 'graduate'
    'deployment_status': 'deprecated'->  'maturity': 'retired'
    'deployment_status': 'baseline'  ->  'maturity': 'candidate'  (+ baseline is a role)

`config_deployment.py` is still the field the code reads. Nothing reads this file yet; the
rename cannot ride the pipeline-core 3.0 release, so setting this is a declaration of
intent ahead of Phase 2, not a switch.

**Why `candidate` on a finished baseline.** ADR-017 uses `graduate` for two jobs at once —
"finished" and "eligible to ship" — and they disagree here: a zero/average/locf baseline is
complete, but must never be sent to a partner. No value in the closed set is true for it.
`candidate` is the truthful translation of the `shadow` this model already carried, and it
asserts nothing about shelf eligibility. Resolving the ADR gap is tracked separately.
"""

def get_maturity_config():
    # Maturity settings
    maturity_config = {'maturity': 'candidate'}
    return maturity_config
