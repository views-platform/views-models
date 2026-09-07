"""
Maturity Configuration Script — ADR-017 Axis 1.

**This is the reference example for the `deployment_status` -> `maturity` migration
(ADR-017 §3, Phase 2).** Copy this shape exactly; do not invent variations.

Maturity answers ONE question: how finished is this source? It says nothing about
where the forecast goes (that is a delivery, ADR-019) and nothing about what an
ensemble contains (that is `config_modelset.py`).

Values — a closed set of exactly three:
- candidate: in development. Not finished, not to be run for production purposes.
- graduate:  finished. Ready to be run, selected on, and eligible to ship.
- retired:   dead. No active ensemble may contain a retired member (ADR-017 §5, R1).

`baseline` is NOT a maturity and does not appear here. It is a *role* — a naive
yardstick you score against — already carried by the algorithm and by
`regression_point_baselines` in `config_meta.py` (ADR-017 §3).

**During the transition BOTH files are present and must agree** (ADR-017 §11, the
dual-vocabulary window). The mapping is:

    config_deployment.py            config_maturity.py
    'deployment_status': 'shadow'      'maturity': 'candidate'
    'deployment_status': 'deployed'    'maturity': 'graduate'
    'deployment_status': 'deprecated'  'maturity': 'retired'
    'deployment_status': 'baseline'    'maturity': 'candidate'   (+ it is a baseline by role)

Nothing reads this file yet. The rename cannot ride the pipeline-core 3.0 release,
so `config_deployment.py` remains the field the code actually reads until Phase 2
completes. Setting this file now is how a source declares its intent ahead of that.
"""

def get_maturity_config():
    # Maturity settings
    maturity_config = {'maturity': 'candidate'}
    return maturity_config
