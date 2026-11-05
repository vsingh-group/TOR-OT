"""UNOT adapter (env_unot):  pytest baselines/tests/test_unot.py"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("wandb")
from baselines.common import cost_matrix  # noqa: E402
from baselines.unot import train as ut  # noqa: E402


def test_generator1d_outputs_histograms():
    hp = dict(ut.DEFAULTS, width=32)
    gen, _ = ut.build(30, hp, torch.device("cpu"))
    mu, nu = gen(torch.randn(16, 2 * hp["length_latent"]))
    for x in (mu, nu):
        assert x.shape == (16, 30) and (x > 0).all() and torch.allclose(x.sum(1), torch.ones(16))


def test_official_training_loop_runs_and_plans_match_source_marginal():
    import wandb
    wandb.init(mode="disabled")
    n, eps, hp = 30, 0.05, dict(ut.DEFAULTS, width=32, nits=3, batch_size=8)
    C = cost_matrix(n, "sqeuclidean")
    gen, pred = ut.build(n, hp, torch.device("cpu"))
    trainer = ut.Training(gen, pred, lr=hp["lr"], weight_decay_generator=0.0, weight_decay_predictor=1e-4,
                          multiplicative_factor_generator=1.0, multiplicative_factor_predictor=0.9999,
                          numbr_training_iterations=hp["nits"], numbr_mini_loop_predictor=1, numbr_mini_loop_generator=1,
                          sinkhorn_max_iterations=5, sinkhorn_epsilon=eps, cost_matrix=torch.as_tensor(C, dtype=torch.float32),
                          batch_size=hp["batch_size"], dim_prior=hp["length_latent"])
    trainer.train()
    assert len(trainer.train_losses["predictor"]) == hp["nits"]
    rng = np.random.default_rng(0)
    a, b = rng.dirichlet(np.ones(n), 4), rng.dirichlet(np.ones(n), 4)
    P = ut.plans_from_predictor(pred, a, b, C, eps, torch.device("cpu"))
    assert P.shape == (4, n, n) and np.isfinite(P).all()
    np.testing.assert_allclose(P.sum(2), a, atol=1e-6)        # u = a / (K v): row marginals exact


def test_log_domain_targets_match_the_official_ones():
    """At an eps where the official float32 scaling-domain Sinkhorn does not underflow, both give the same targets."""
    import types
    import wandb
    wandb.init(mode="disabled")
    n, eps, hp = 30, 0.05, dict(ut.DEFAULTS, width=32)
    C = torch.as_tensor(cost_matrix(n, "sqeuclidean"), dtype=torch.float32)
    gen, pred = ut.build(n, hp, torch.device("cpu"))
    tr = ut.Training(gen, pred, lr=0.01, weight_decay_generator=0.0, weight_decay_predictor=1e-4, multiplicative_factor_generator=1.0,
                     multiplicative_factor_predictor=0.9999, numbr_training_iterations=1, numbr_mini_loop_predictor=1,
                     numbr_mini_loop_generator=1, sinkhorn_max_iterations=5, sinkhorn_epsilon=eps, cost_matrix=C, batch_size=8,
                     dim_prior=hp["length_latent"])
    with torch.no_grad():
        mu, nu = gen(torch.randn(8, 2 * hp["length_latent"]))
    g_official, m_official = tr.target_sinkhorn(mu, nu)
    g_log, m_log = types.MethodType(ut.log_domain_targets, tr)(mu, nu)
    assert m_official.all() and m_log.all()
    centred = lambda g: g - g.mean(1, keepdim=True)              # potentials are defined up to a constant
    torch.testing.assert_close(centred(g_log), centred(g_official), atol=1e-3, rtol=1e-3)
