"""Marginals (histograms) over bin coordinates.

Training / in-distribution marginals are drawn by one of the ``MARGINAL_SAMPLERS``; no
parametric family is assumed:

* ``simplex`` (default): uniform on the probability simplex, i.e. Dirichlet(1, ..., 1)
  (equivalently: i.i.d. Exponential(1) weights, normalised).
* ``normalized_uniform``: i.i.d. U(0, 1) weights normalised to sum to one. This is what v1
  and the ``torot_data_1M_v2`` corpus used (labelled ``uniform_random`` there). It is *not*
  uniform on the simplex: it concentrates near the centre (histograms stay close to flat; a bin
  essentially never exceeds 3/n, which ~5 % of bins do under Dirichlet(1) at n = 30).

The named 1D densities below are used **only** for the out-of-distribution evaluation
split, mirroring the OOD evaluation of the paper.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
from scipy import stats

MIN_MASS = 1e-6  # bins below this mass are zeroed before renormalisation (numerical hygiene)


def _normalise(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.where(np.isfinite(p) & (p > 0.0), p, 0.0)
    p[p < MIN_MASS * max(p.sum(), 1e-300)] = 0.0
    total = p.sum()
    if total <= 0.0:
        p = np.ones_like(p)
        total = p.sum()
    return p / total


def uniform_simplex(rng: np.random.Generator, n: int) -> np.ndarray:
    """Uniform on the probability simplex (Dirichlet(1, ..., 1)).

    No bin is zeroed: tiny masses are part of this distribution and are kept as drawn.
    """
    return rng.dirichlet(np.ones(n))


def normalized_uniform(rng: np.random.Generator, n: int) -> np.ndarray:
    """v1 / ``torot_data_1M_v2`` marginal: i.i.d. U(0, 1) weights over ``n`` bins, normalised."""
    return _normalise(rng.uniform(size=n))


MARGINAL_SAMPLERS: Dict[str, Callable[[np.random.Generator, int], np.ndarray]] = {
    "simplex": uniform_simplex,
    "normalized_uniform": normalized_uniform,
}


# --------------------------------------------------------------------------- named 1D densities (OOD evaluation only)
def _pdf_on_unit(dist) -> Callable[[np.ndarray], np.ndarray]:
    return lambda x: dist.pdf(x)


NAMED_1D_FAMILIES: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "beta_left": _pdf_on_unit(stats.beta(2.0, 5.0)),
    "beta_right": _pdf_on_unit(stats.beta(5.0, 2.0)),
    "lognormal": lambda x: stats.lognorm(s=0.5).pdf(4.0 * x),
    "laplace": _pdf_on_unit(stats.laplace(loc=0.5, scale=0.12)),
    "student_t": _pdf_on_unit(stats.t(df=3, loc=0.5, scale=0.12)),
    "cauchy": _pdf_on_unit(stats.cauchy(loc=0.5, scale=0.08)),
    "triangular": _pdf_on_unit(stats.triang(c=0.5)),
    "uniform_mid": _pdf_on_unit(stats.uniform(loc=0.25, scale=0.5)),
    "gaussian_shifted": _pdf_on_unit(stats.norm(loc=0.65, scale=0.12)),
    "exponential": _pdf_on_unit(stats.expon(scale=0.25)),
    "weibull": _pdf_on_unit(stats.weibull_min(1.5, scale=0.4)),
    "gamma": _pdf_on_unit(stats.gamma(2.0, scale=0.15)),
}


def named_marginal(name: str, coords: np.ndarray) -> np.ndarray:
    """Deterministic histogram of a named 1D density evaluated at the bin centres."""
    if coords.shape[1] != 1:
        raise ValueError("named families are defined for 1D supports only")
    return _normalise(NAMED_1D_FAMILIES[name](coords[:, 0]))
