"""Meta OT adapter (env_metaot):  pytest baselines/tests/test_metaot.py"""
import os

import numpy as np
import pytest

pytest.importorskip("ott")
import jax  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from baselines.common import cost_matrix  # noqa: E402
from baselines.metaot import train as mt  # noqa: E402


def small_cfg(iters):
    cfg = OmegaConf.load(mt.CONF)
    cfg.pop("hydra", None)
    cfg.data, cfg.seed, cfg.num_train_iter, cfg.batch_size = "random", 0, iters, 16
    cfg.potential_model_mlp.n_hidden, cfg.potential_model_mlp.n_hidden_layer = 32, 2
    return cfg


def test_official_training_loop_runs_and_plans_match_target_marginal(tmp_path):
    n, eps = 6, 0.05
    C = cost_matrix(n, "sqeuclidean")
    mt.meta_data.RandomSampler = mt.simplex_sampler(C, eps)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        ws = mt.Workspace1D(small_cfg(101), n, C, eps)
        ws.run()
    finally:
        os.chdir(cwd)
    assert (tmp_path / "log.csv").exists() and (tmp_path / "latest.pkl").exists()
    rng = np.random.default_rng(0)
    a, b = rng.dirichlet(np.ones(n), 3), rng.dirichlet(np.ones(n), 3)
    P = ws.predict(a, b)
    assert P.shape == (3, n, n) and np.isfinite(P).all()
    np.testing.assert_allclose(P.sum(1), b, atol=1e-8)        # g = c-transform of f: column marginals exact


def test_sampler_draws_simplex_pairs():
    batch = mt.simplex_sampler(cost_matrix(5, "euclidean"), 0.01)(batch_size=8)(jax.random.PRNGKey(0))
    a, b = np.asarray(batch.a), np.asarray(batch.b)
    assert a.shape == b.shape == (8, 5) and (a > 0).all() and np.allclose(a.sum(1), 1) and np.allclose(b.sum(1), 1)
