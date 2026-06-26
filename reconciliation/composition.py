"""Wire a reconciler for an ensemble run (EPIC #172 / S4 #177).

The composition logic a reconciling ensemble's ``main.py`` calls: derive the
forecast month window from the ensemble's partition config and build the
reconciler (geography source **derived** from the ensemble's data — viewser vs
datafactory). Keeps ``main.py`` a thin one-liner; window-sizing and source
derivation live here (SRP), not in the leaf.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional

from views_pipeline_core.domain.reconciliation import Reconciler

from reconciliation.country_mapping_provider import CountryMappingProvider
from reconciliation.reconciler_factory import build_reconciler
from reconciliation.source_detection import detect_ensemble_source

# Months padded past the last declared test range, so a forecast run that runs
# beyond the fixed partitions is still covered by the geography mapping.
_WINDOW_BUFFER_MONTHS = 24


def _forecast_window(partitions: dict, buffer: int = _WINDOW_BUFFER_MONTHS) -> tuple[int, int]:
    """The (start, end) month span the geography must cover — the union of every
    partition's test range, padded. A superset is safe; a missing month is not."""
    test_ranges = [
        p["test"] for p in partitions.values() if isinstance(p, dict) and "test" in p
    ]
    if not test_ranges:
        raise ValueError(
            "config_partitions declares no test ranges; cannot size the reconciliation window"
        )
    starts = [int(r[0]) for r in test_ranges]
    ends = [int(r[1]) for r in test_ranges]
    return min(starts), max(ends) + buffer


def _load_partitions(ensemble_dir: Path) -> dict:
    path = Path(ensemble_dir) / "configs" / "config_partitions.py"
    spec = importlib.util.spec_from_file_location(
        f"_recon_partitions_{Path(ensemble_dir).name}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate()


def _load_meta(ensemble_dir: Path) -> dict:
    path = Path(ensemble_dir) / "configs" / "config_meta.py"
    spec = importlib.util.spec_from_file_location(
        f"_recon_meta_{Path(ensemble_dir).name}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_meta_config()


def _derive_source(ensemble_dir: Path) -> str:
    """Derive the geography source from the data being reconciled (EPIC #192).

    The country-id system must match the CM forecast this ensemble reconciles
    against (its ``reconcile_with`` partner — VIEWS ``country_id`` vs ``gaul0_code``).
    The PGM ensemble's own source must agree, or the pairing is incoherent. Fails
    loud on mismatch; an unsupported source (e.g. datafactory before its provider
    exists) then fails loud at the factory — never a silent viewser fallback.
    """
    ensemble_dir = Path(ensemble_dir)
    partner = _load_meta(ensemble_dir).get("reconcile_with")
    if not partner:
        raise ValueError(
            f"{ensemble_dir.name}: reconciliation is configured but no reconcile_with "
            f"partner is declared — cannot derive the geography source."
        )
    cm_source = detect_ensemble_source(ensemble_dir.parent / partner)
    pgm_source = detect_ensemble_source(ensemble_dir)
    if pgm_source != cm_source:
        raise ValueError(
            f"{ensemble_dir.name} (source={pgm_source}) and its reconcile_with partner "
            f"'{partner}' (source={cm_source}) disagree on data source — reconciliation "
            f"would mix country-id systems. Both must share one source (C-49, EPIC #192)."
        )
    return cm_source


def build_reconciler_for_run(
    ensemble_dir: Path,
    source: Optional[str] = None,
    provider: Optional[CountryMappingProvider] = None,
) -> Reconciler:
    """Build the reconciler for a reconciling ensemble. Sizes the geography window
    from the partition config and **derives** the geography source from the data
    (via the ``reconcile_with`` CM partner) unless an explicit ``source``/``provider``
    is given. A datafactory-sourced ensemble fails loud at the factory until its
    provider exists — never a silent viewser fallback (EPIC #192 / ADR-014)."""
    ensemble_dir = Path(ensemble_dir)
    start_month, end_month = _forecast_window(_load_partitions(ensemble_dir))
    if provider is not None:
        return build_reconciler(start_month, end_month, provider=provider)
    if source is None:
        source = _derive_source(ensemble_dir)
    return build_reconciler(start_month, end_month, source=source)
