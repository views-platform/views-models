"""Probe: the time coverage of the injected mapping vs the forecast horizon. Read-only."""
import sys
from pathlib import Path
import numpy as np

REPO = Path("/workspace/views-models")
sys.path.insert(0, str(REPO))

from reconciliation.composition import _forecast_window, _load_partitions
from reconciliation.reconciler_factory import build_reconciler
from reconciliation.datafactory_country_mapping_provider import (
    DatafactoryCountryMappingProvider,
)

d = REPO / "ensembles" / "chrome_box"
start, end = _forecast_window(_load_partitions(d))
print(f"window ASKED FOR: {start}..{end}")

p = DatafactoryCountryMappingProvider(start, end)
m = p.build()
times = m.map_keys[:, 0]
gids = m.map_keys[:, 1]
print(f"mapping GOT:      {times.min()}..{times.max()}  "
      f"({np.unique(times).size} months x {np.unique(gids).size} cells = {len(m)} keys)")
print(f"MISSING MONTHS:   {int(times.max())+1}..{end}  "
      f"({end - int(times.max())} of {end - start + 1})")
