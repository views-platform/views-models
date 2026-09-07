"""
Maturity Configuration Script — ADR-017 Axis 1.

**Maturity answers exactly one question: has the author signed off that this model is done
and works as expected?**

It is not a judgement of whether the model is good, whether it beats a baseline, or whether
it adds value to any particular ensemble. It is a statement of workmanship by the person who
built it.

It is also NOT a decision to ship. Shipping is a separate axis — the delivery declaration
(ADR-019), which names sources per consumer. ADR-017 §4c splits the two deliberately so the
producer never knows its consumers, and §4d says plainly: *"A graduate-but-undelivered
forecast sitting there is fine: it is finished, just not routed anywhere yet."*

This matters because you cannot know in advance which models end up in a shipped ensemble.
A model can be worth including in one ensemble and not another. So membership is decided per
ensemble, not encoded here.

Values — a closed set of exactly three:
- candidate: not signed off. Still being worked on.
- graduate:  the author signs off — done, works as expected. Eligible to be considered for
             any ensemble, and eligible to reach the production shelf.
- retired:   do not use. No active ensemble may contain a retired member (ADR-017 §5, R1).

`baseline` is NOT a maturity and does not appear here. It is a *role* — a naive yardstick you
score against — carried by `regression_point_baselines` / `regression_sample_baselines` in
`config_meta.py`. A baseline can perfectly well be `graduate`: it is finished and works as
expected. Nothing ships it, because no delivery names it.

**Both config files exist during the transition and must agree** (ADR-017 §11, the
dual-vocabulary window):

    config_deployment.py                  config_maturity.py
    'deployment_status': 'shadow'     ->  'maturity': 'candidate'
    'deployment_status': 'deployed'   ->  'maturity': 'graduate'
    'deployment_status': 'deprecated' ->  'maturity': 'retired'

`config_deployment.py` is still the field the code reads; nothing reads this file yet. The
rename cannot ride the pipeline-core 3.0 release (ADR-017 §11 Phase 2), so setting this is a
declaration of intent ahead of that, not a switch.
"""

def get_maturity_config():
    # Maturity settings
    maturity_config = {'maturity': 'graduate'}
    return maturity_config
