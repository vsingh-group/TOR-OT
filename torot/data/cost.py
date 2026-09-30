"""Ground-cost matrices. The cost is identified by its *type* only; the model never sees C."""

from __future__ import annotations

import numpy as np
import ot

COST_TYPES: tuple[str, ...] = (
    "sqeuclidean",
    "euclidean",
    "cityblock",
    "chebyshev",
    "minkowski",
    "canberra",
)
MINKOWSKI_P = 3


def cost_matrix(xs: np.ndarray, xt: np.ndarray, cost_type: str) -> np.ndarray:
    """Return the max-normalised cost matrix C[i, j] = c(xs[i], xt[j]) with C.max() == 1.

    All six metrics are symmetric, c(x, y) = c(y, x), which is what the transpose
    augmentation relies on.
    """
    if cost_type not in COST_TYPES:
        raise ValueError(f"unknown cost type {cost_type!r}; expected one of {COST_TYPES}")
    xs = np.asarray(xs, dtype=np.float64)
    xt = np.asarray(xt, dtype=np.float64)
    C = ot.dist(xs, xt, metric=cost_type, p=MINKOWSKI_P)
    C = np.asarray(C, dtype=np.float64)
    cmax = C.max()
    if cmax > 0:
        C = C / cmax
    return C
