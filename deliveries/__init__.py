"""Forecast deliveries — one file per consumer (ADR-017 §3, ADR-019).

`deliveries/<consumer>.py` is the only place a delivery is declared. The **filename is
the consumer**, so no key repeats it and the two cannot disagree.

A source never mentions a consumer; a delivery never sets a maturity. "Is this in
production?" is derived from the two together (ADR-017 §4e), never typed.
"""
