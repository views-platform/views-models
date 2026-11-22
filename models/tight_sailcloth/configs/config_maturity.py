"""
Maturity Configuration Script — ADR-017 Axis 1.

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

A new source starts as `candidate`, and this file is where it is promoted. Maturity is
earned: the scaffolder cannot set it to anything else, deliberately — a source is not
finished the moment it is created, and nothing else in the platform is in a position to
say that it is.

This file is the only place this source declares its maturity. It is read by
`views_pipeline_core`'s config loader and validated on every run.
"""

def get_maturity_config():
    # Maturity settings
    maturity_config = {'maturity': 'candidate'}
    return maturity_config
