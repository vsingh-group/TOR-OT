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


# --------------------------------------------------------------------------- named 2D distributions (OOD evaluation only)
# The seven 2D distributions of the submission's 2D evaluation (TF4OT/run_inference_2d.py,
# eval_all_2d_distribution_pairs), with the same parameters: point clouds standardised per axis (except the standard
# normal) and binned on [-2.5, 2.5]^2. The scikit-learn generators (make_swiss_roll / make_moons / make_circles /
# make_s_curve) are re-implemented with NumPy from their definitions, so no scikit-learn dependency is needed.
NAMED_2D_RANGE = (-2.5, 2.5)
NAMED_2D_SAMPLES = 100_000


def _standardize(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(0, keepdims=True)
    std = x.std(0, ddof=1, keepdims=True)
    return x / np.where(std == 0, 1.0, std)


def _swiss_roll(rng, n, noise=0.8):
    t = 1.5 * np.pi * (1 + 2 * rng.random(n))
    x, y, z = t * np.cos(t), 21 * rng.random(n), t * np.sin(t)
    pts = np.stack([x, y, z], 1) + noise * rng.standard_normal((n, 3))
    return _standardize(pts[:, [0, 2]])


def _mix8_gaussians(rng, n, std=0.5, r=12.0):
    s = 1 / np.sqrt(2)
    centers = r * np.array([(1, 0), (-1, 0), (0, 1), (0, -1), (s, s), (s, -s), (-s, s), (-s, -s)])
    return _standardize(centers[rng.integers(0, 8, n)] + std * rng.standard_normal((n, 2)))


def _standard_normal(rng, n):
    return rng.standard_normal((n, 2))


def _moons(rng, n, noise=0.1):
    n_out = n // 2
    n_in = n - n_out
    t_out, t_in = np.linspace(0, np.pi, n_out), np.linspace(0, np.pi, n_in)
    pts = np.concatenate([np.stack([np.cos(t_out), np.sin(t_out)], 1),
                          np.stack([1 - np.cos(t_in), 1 - np.sin(t_in) - 0.5], 1)])
    return _standardize(pts + noise * rng.standard_normal((n, 2)))


def _circles(rng, n, noise=0.05, factor=0.5):
    n_out = n // 2
    n_in = n - n_out
    t_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    t_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False)
    pts = np.concatenate([np.stack([np.cos(t_out), np.sin(t_out)], 1),
                          factor * np.stack([np.cos(t_in), np.sin(t_in)], 1)])
    return _standardize(pts + noise * rng.standard_normal((n, 2)))


def _s_curve(rng, n, noise=0.1):
    t = 3 * np.pi * (rng.random(n) - 0.5)
    x, y, z = np.sin(t), 2.0 * rng.random(n), np.sign(t) * (np.cos(t) - 1)
    pts = np.stack([x, y, z], 1) + noise * rng.standard_normal((n, 3))
    return _standardize(pts[:, [0, 2]])


def _uniform_square(rng, n, low=-2.0, high=2.0):
    return _standardize(rng.random((n, 2)) * (high - low) + low)


NAMED_2D_FAMILIES: Dict[str, Callable[[np.random.Generator, int], np.ndarray]] = {
    "SwissRoll": _swiss_roll,
    "Mix8Gaussians": _mix8_gaussians,
    "StandardNormal": _standard_normal,
    "Moons": _moons,
    "Circles": _circles,
    "SCurve": _s_curve,
    "UniformSquare": _uniform_square,
}


def named_marginal_2d(name: str, side: int) -> np.ndarray:
    """Deterministic histogram of a named 2D distribution on the ``side x side`` grid of ``supports.grid_2d``: a fixed
    seed per family, NAMED_2D_SAMPLES points binned on NAMED_2D_RANGE^2, flattened x-major. Mass outside the range is
    dropped (as in the submission); empty bins stay empty."""
    rng = np.random.default_rng(7_000_000 + list(NAMED_2D_FAMILIES).index(name))
    pts = NAMED_2D_FAMILIES[name](rng, NAMED_2D_SAMPLES)
    counts, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[side, side], range=[NAMED_2D_RANGE, NAMED_2D_RANGE])
    return counts.ravel() / counts.sum()
