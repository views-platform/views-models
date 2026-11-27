"""The reconciler factory — the single concrete-binding wire (EPIC #172 / S3 #176).

This is the **only** file that imports the concrete reconciler
(`views_frames_reconcile.ReconciliationModule`) — the one sanctioned, irreducible
composition wire (ADR-014; CCP/DIP). The concrete moved from
`views_postprocessing.reconciliation` to the frames-native `views_frames_reconcile`
sibling in the Epic 11 cutover (views-frames ADR-023, #191). It selects a geography provider,
builds the `(time, priogrid_gid) -> country_id` mapping, constructs the concrete,
and returns it typed as the pipeline-core `Reconciler` port so callers depend only
on the abstraction.

That OCP claim was tested by #601, which added the datafactory provider: it cost one
new provider file and one registry entry, with no change to this function's signature
or to any caller. A third source costs the same.
"""
from __future__ import annotations

from typing import Optional

from views_pipeline_core.domain.reconciliation_port import Reconciler

from reconciliation.country_mapping_provider import CountryMappingProvider
from reconciliation.datafactory_country_mapping_provider import (
    DatafactoryCountryMappingProvider,
)
from reconciliation.source_detection import DATAFACTORY, VIEWSER
from reconciliation.viewser_country_mapping_provider import ViewserCountryMappingProvider

# Geography-source name -> provider class. Selected per ensemble from its data
# source (ADR-013); callers don't change when one is added (OCP).
#
# ⛔ The keys are `source_detection`'s own constants, imported rather than retyped.
# This comment used to say to add `"datafactory"`, and #601 records why that would
# have been wrong: the key must equal what `detect_ensemble_source` RETURNS, which is
# `"views-datafactory"`. A literal here can drift from the detector silently, and the
# symptom would be the same `ValueError` below — "unknown source" for a source that
# had just been added.
_PROVIDERS = {
    VIEWSER: ViewserCountryMappingProvider,
    DATAFACTORY: DatafactoryCountryMappingProvider,
}


def provider_for(
    source: str, start_month: int, end_month: int
) -> CountryMappingProvider:
    """The geography provider for ``source``, covering ``start_month..end_month``.

    Named and public so that a *verifier* of a finished run can build the identical
    mapping the run used. Before this existed, the only way to do that was to reach into
    ``_PROVIDERS`` from outside, which makes sameness a coincidence rather than a
    guarantee — and a verifier that silently measures against a *different* mapping
    reports a clean verdict it never established (ADR-025).
    """
    try:
        provider_cls = _PROVIDERS[source]
    except KeyError:
        raise ValueError(
            f"Unknown reconciliation geography source {source!r}; "
            f"known sources: {sorted(_PROVIDERS)}"
        )
    return provider_cls(start_month, end_month)


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
        provider = provider_for(source, start_month, end_month)

    mapping = provider.build()

    # The single concrete import — confined to this file (ADR-014). Frames-native
    # since the Epic 11 cutover (views-frames ADR-023, #191).
    from views_frames_reconcile import ReconciliationModule

    return ReconciliationModule(mapping.map_keys, mapping.map_vals)
