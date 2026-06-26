"""Reconciliation composition layer for views-models (EPIC #172 / ADR-014).

The single sanctioned place that wires the reconciler Dependency-Inversion seam:
pipeline-core defines the `Reconciler` port, views-postprocessing provides the
concrete `ReconciliationModule`, and this layer (the composition root's helper)
builds the geography and constructs the concrete — confined to one file.

The composition root imports `build_reconciler` and injects its result as
`reconciler=` into the ensemble manager.
"""
from reconciliation.country_mapping import CountryMapping
from reconciliation.country_mapping_provider import CountryMappingProvider
from reconciliation.reconciler_factory import build_reconciler

__all__ = ["CountryMapping", "CountryMappingProvider", "build_reconciler"]
