"""The geography mapping value for reconciliation (EPIC #172 / ADR-014).

A `CountryMapping` is the injected `(time, priogrid_gid) -> country_id` mapping the
reconciler needs. Geography is *injected*, never embedded in the reconciler
(views-frames ADR-014); this immutable value object only holds and validates it.
The country-id system (VIEWS `country_id` vs datafactory `gaul0_code`) is the
provider's concern — this value is agnostic to which one it carries.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CountryMapping:
    """Immutable `(time, priogrid_gid) -> country_id` mapping.

    - ``map_keys``: ``(M, 2)`` int array; row ``i`` is ``(time_i, priogrid_gid_i)``.
    - ``map_vals``: ``(M,)`` int array; ``map_vals[i]`` is the ``country_id`` for
      ``map_keys[i]`` (1-to-1, same order).

    Shapes line up with `views_frames_reconcile.ReconciliationModule`,
    which is constructed as ``ReconciliationModule(map_keys, map_vals)``.
    """

    map_keys: NDArray[np.integer]
    map_vals: NDArray[np.integer]

    def __post_init__(self) -> None:
        keys = np.asarray(self.map_keys)
        vals = np.asarray(self.map_vals)
        if keys.ndim != 2 or keys.shape[1] != 2:
            raise ValueError(
                f"map_keys must be a (M, 2) array of (time, priogrid_gid); got shape {keys.shape}"
            )
        if vals.ndim != 1 or vals.shape[0] != keys.shape[0]:
            raise ValueError(
                f"map_vals must be a (M,) array aligned 1-to-1 with map_keys "
                f"(M={keys.shape[0]}); got shape {vals.shape}"
            )
        if not np.issubdtype(keys.dtype, np.integer) or not np.issubdtype(vals.dtype, np.integer):
            raise ValueError(
                f"map_keys/map_vals must be integer arrays; got {keys.dtype} / {vals.dtype}"
            )
        # frozen dataclass: store the coerced arrays.
        object.__setattr__(self, "map_keys", keys)
        object.__setattr__(self, "map_vals", vals)

    def __len__(self) -> int:
        return int(self.map_keys.shape[0])
