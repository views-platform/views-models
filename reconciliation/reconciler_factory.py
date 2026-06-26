"""The reconciler factory — the single concrete-binding wire (EPIC #172 / S3 #176).

This is the **only** file that imports the concrete reconciler
(`views_postprocessing.reconciliation.ReconciliationModule`) — the one sanctioned,
irreducible composition wire (ADR-014; CCP/DIP). It selects a geography provider,
builds the `(time, priogrid_gid) -> country_id` mapping, constructs the concrete,
and returns it typed as the pipeline-core `Reconciler` port so callers depend only
on the abstraction.

Adding a datafactory-sourced ensemble later is a one-line registry entry +
one new provider file — no change to this function's signature or its callers (OCP).
"""
from __future__ import annotations

from typing import Optional

from views_pipeline_core.domain.reconciliation import Reconciler

from reconciliation.country_mapping_provider import CountryMappingProvider
from reconciliation.viewser_country_mapping_provider import ViewserCountryMappingProvider

# Geography-source name -> provider class. Selected per ensemble from its data
# source (ADR-013). Add "datafactory" here (one line) when a datafactory-sourced
# reconciling ensemble first exists — callers don't change (OCP).
_PROVIDERS = {"viewser": ViewserCountryMappingProvider}


def build_reconciler(
    start_month: int,
    end_month: int,
    source: str = "viewser",
    provider: Optional[CountryMappingProvider] = None,
) -> Reconciler:
    """Construct the concrete reconciler for a forecast window, typed as the port.

    Args:
        start_month, end_month: the forecast month range the mapping must cover.
        source: geography-source key (default ``"viewser"``); selects the provider.
        provider: an explicit provider (overrides ``source``) — for testing / advanced wiring.
    """
    if provider is None:
        try:
            provider_cls = _PROVIDERS[source]
        except KeyError:
            raise ValueError(
                f"Unknown reconciliation geography source {source!r}; "
                f"known sources: {sorted(_PROVIDERS)}"
            )
        provider = provider_cls(start_month, end_month)

    mapping = provider.build()

    # The single concrete import — confined to this file (ADR-014).
    from views_postprocessing.reconciliation import ReconciliationModule

    return ReconciliationModule(mapping.map_keys, mapping.map_vals)
