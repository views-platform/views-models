"""
Maturity Configuration Script

This script declares the model's maturity (ADR-057), replacing the legacy
`deployment_status` vocabulary.

Maturity values:
- candidate: under development; validated but not part of production
  (successor of the legacy 'shadow').
- graduate: fully developed and in production use.
- retired: superseded or no longer maintained.

"""

def get_maturity_config():
    # Maturity settings
    maturity_config = {'maturity': 'candidate'}
    return maturity_config
