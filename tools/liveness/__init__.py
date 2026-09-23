"""Liveness checks: is each forecast surface live, fresh, and serving?

One module per external surface, each answering with RAW FACTS (URL hit,
value found, date derived) and a verdict — never narration. Run a single
surface as ``python -m tools.liveness.<surface>``; the aggregate runner
arrives with epic #238 story S7.

Surfaces (epic #238): old_api (S1, here) · datafactory input (S2) ·
Appwrite production_forecasts (S3) · FAO unfao_bucket (S4) · wandb
execution (S5) · VPN store gjoll (S6).

House rules: zero import-time side effects (C-93); injected fetch for
testability (DIP, mirroring reconciliation/viewser_country_mapping_provider);
zero new dependencies; live tests skip truthfully (C-75).
"""
