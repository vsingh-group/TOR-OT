"""Ground-cost matrices. The cost is identified by its *type* only; the model never sees C.

The six types are chosen to be pairwise distinct on a 1D support (on a line L1, L2, L-inf
and Minkowski all collapse to |x - y|), all symmetric, c(x, y) = c(y, x), and to cover
different geometries: strictly convex (unique optimum, monotone plans), linear
(non-unique optima), concave, bounded/saturating, and a non-translation-invariant one.
"""

from __future__ import annotations

import numpy as np
import ot

COST_TYPES: tuple[str, ...] = (
    "sqeuclidean",      # |x - y|^2          strictly convex
    "euclidean",        # |x - y|            linear: = L1 = L-inf in 1D, optimum generally non-unique
    "canberra",         # sum_d |x_d - y_d| / (|x_d| + |y_d|)   not translation invariant
    "sqrt_euclidean",   # |x - y|^0.5        concave
    "cubic_euclidean",  # |x - y|^3          strictly convex, heavier penalty on long moves
    "gaussian",         # 1 - exp(-|x - y|^2 / (2 s^2)), s = 0.2   bounded, saturating
)
GAUSSIAN_SCALE = 0.2


def cost_matrix(xs: np.ndarray, xt: np.ndarray, cost_type: str) -> np.ndarray:
    """Return the max-normalised cost matrix C[i, j] = c(xs[i], xt[j]) with C.max() == 1."""
    if cost_type not in COST_TYPES:
        raise ValueError(f"unknown cost type {cost_type!r}; expected one of {COST_TYPES}")
    xs = np.asarray(xs, dtype=np.float64)
    xt = np.asarray(xt, dtype=np.float64)
    if cost_type == "canberra":
        C = np.asarray(ot.dist(xs, xt, metric="canberra"), dtype=np.float64)
    else:
        dist = np.sqrt(np.maximum(ot.dist(xs, xt, metric="sqeuclidean"), 0.0))
        if cost_type == "sqeuclidean":
            C = dist**2
        elif cost_type == "euclidean":
            C = dist
        elif cost_type == "sqrt_euclidean":
            C = np.sqrt(dist)
        elif cost_type == "cubic_euclidean":
            C = dist**3
        else:  # gaussian
            C = 1.0 - np.exp(-(dist**2) / (2.0 * GAUSSIAN_SCALE**2))
    cmax = C.max()
    if cmax > 0:
        C = C / cmax
    return C
