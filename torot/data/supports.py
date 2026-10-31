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


def grid_2d(k: int) -> np.ndarray:
    """``k x k`` bin centres of a regular grid on [0, 1]^2, shape [k*k, 2], in canonical order: x-major (index
    i * k + j is the bin (x_i, y_j)), the row-major order of ``np.histogram2d`` and of the submission's 2D grids."""
    c = (np.arange(k, dtype=np.float64) + 0.5) / k
    xx, yy = np.meshgrid(c, c, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel()], axis=1)


def grid(n_side: int, dim: int) -> np.ndarray:
    """Regular grid of ``n_side`` bins per axis in ``dim`` dimensions (1D: n_side bins; 2D: n_side^2 bins)."""
    if dim == 1:
        return grid_1d(n_side)
    if dim == 2:
        return grid_2d(n_side)
    raise ValueError(f"dim must be 1 or 2, got {dim}")


def set_coords(z) -> np.ndarray:
    """Bin coordinates of an evaluation set (``reproduce/build_sets*.py``): the stored ``coords`` (2D sets), else the
    1D grid of ``n`` bins (the 1D sets store only ``n``)."""
    if "coords" in z:
        return np.asarray(z["coords"], dtype=np.float64)
    return grid_1d(int(z["n"]))


def pad_coords(coords: np.ndarray, dim_out: int = COORD_DIM) -> np.ndarray:
    """Right-pad coordinates with zeros to ``dim_out`` columns (1D instances become (x, 0))."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[1] == dim_out:
        return coords
    out = np.zeros((coords.shape[0], dim_out), dtype=np.float64)
    out[:, : coords.shape[1]] = coords
    return out
