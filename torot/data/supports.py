"""Support (bin coordinate) generators. Coordinates live in [0, 1].

Bins are returned in *canonical order* (increasing coordinate). The plan serialisation
(rows by source index, entries by target index) relies on this order, which makes the
tokenised sequence a deterministic function of the instance.

Coordinates are carried with two columns so that 2D supports can be added later
without changing the model; 1D instances use (x, 0).
"""

from __future__ import annotations

import numpy as np

COORD_DIM = 2


def grid_1d(n: int) -> np.ndarray:
    """``n`` equally spaced bin centres on [0, 1], shape [n, 1]."""
    return ((np.arange(n, dtype=np.float64) + 0.5) / n)[:, None]


def pad_coords(coords: np.ndarray, dim_out: int = COORD_DIM) -> np.ndarray:
    """Right-pad coordinates with zeros to ``dim_out`` columns (1D instances become (x, 0))."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[1] == dim_out:
        return coords
    out = np.zeros((coords.shape[0], dim_out), dtype=np.float64)
    out[:, : coords.shape[1]] = coords
    return out
