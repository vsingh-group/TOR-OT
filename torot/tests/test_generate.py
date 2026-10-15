import numpy as np
import pytest
from scipy import stats

from torot.data.families import MARGINAL_SAMPLERS, normalized_uniform, uniform_simplex
from torot.data.generate import GenConfig, make_random_instance


def test_simplex_sampler_is_uniform_on_the_simplex():
    """Under Dirichlet(1, ..., 1) every coordinate follows Beta(1, n - 1)."""
    n, rng = 30, np.random.default_rng(0)
    X = np.stack([uniform_simplex(rng, n) for _ in range(4000)])
    assert np.allclose(X.sum(1), 1.0) and (X >= 0).all()
    for j in (0, n // 2, n - 1):
        assert stats.kstest(X[:, j], stats.beta(1, n - 1).cdf).pvalue > 0.01
    assert np.mean(X > 3.0 / n) > 0.03                      # heavy bins occur (~4.7 % expected)


def test_normalized_uniform_is_not_uniform_on_the_simplex():
    n, rng = 30, np.random.default_rng(0)
    X = np.stack([normalized_uniform(rng, n) for _ in range(4000)])
    assert stats.kstest(X[:, 0], stats.beta(1, n - 1).cdf).pvalue < 1e-6
    assert np.mean(X > 3.0 / n) < 1e-3


def test_normalized_uniform_reproduces_the_v1_sampler():
    """Same random stream as v1 / torot_data_1M_v2: normalised i.i.d. U(0, 1) weights."""
    a = normalized_uniform(np.random.default_rng(7), 25)
    w = np.random.default_rng(7).uniform(size=25)
    assert np.allclose(a, w / w.sum())


def test_generator_uses_the_configured_sampler():
    rec = make_random_instance(3, GenConfig(n_min=10, n_max=10))
    assert rec["family_src"] == rec["family_tgt"] == "simplex"
    rec = make_random_instance(3, GenConfig(n_min=10, n_max=10, marginals="normalized_uniform"))
    assert rec["family_src"] == "normalized_uniform"
    assert set(MARGINAL_SAMPLERS) == {"simplex", "normalized_uniform"}
    with pytest.raises(ValueError):
        GenConfig(marginals="uniform_random")
