"""Wire a reconciler for an ensemble run (EPIC #172 / S4 #177).

The composition logic a reconciling ensemble's ``main.py`` calls: derive the
forecast month window from the ensemble's partition config and build the
reconciler (geography source selected per ensemble — viewser today). Keeps
``main.py`` a thin one-liner; the window-sizing lives here (SRP), not in the leaf.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional

from views_pipeline_core.domain.reconciliation import Reconciler

from reconciliation.country_mapping_provider import CountryMappingProvider
from reconciliation.reconciler_factory import build_reconciler

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


def build_reconciler_for_run(
    ensemble_dir: Path,
    source: str = "viewser",
    provider: Optional[CountryMappingProvider] = None,
) -> Reconciler:
    """Build the reconciler for a reconciling ensemble, sizing the geography window
    from its partition config. Called by the ensemble's main.py composition root."""
    start_month, end_month = _forecast_window(_load_partitions(ensemble_dir))
    return build_reconciler(start_month, end_month, source=source, provider=provider)
