"""Empty bins in the random splits (GenConfig.empty_bins_prob): off by default, exact zeros, same instance otherwise."""
import numpy as np
import pytest

from torot.data.generate import GenConfig, make_random_instance


def dense(r):
    P = np.zeros((r["n"], r["m"]))
    P[r["plan_rows"], r["plan_cols"]] = r["plan_vals"]
    return P


def test_off_by_default():
    for dim in (1, 2):
        cfg = GenConfig(dim=dim)
        assert cfg.empty_bins_prob == 0.0
        for seed in range(10):
            r = make_random_instance(seed, cfg)
            assert np.all(np.asarray(r["a"]) > 0) and np.all(np.asarray(r["b"]) > 0)
            assert r["family_src"] == r["family_tgt"] == cfg.marginals


@pytest.mark.parametrize("dim", [1, 2])
def test_every_marginal_emptied_with_prob_one(dim):
    cfg = GenConfig(dim=dim, empty_bins_prob=1.0)
    base = GenConfig(dim=dim)
    patterns = set()
    for seed in range(60):
        r, r0 = make_random_instance(seed, cfg), make_random_instance(seed, base)
        # same support and cost type as without empty bins
        assert (r["n"], r["m"], r["cost_type"]) == (r0["n"], r0["m"], r0["cost_type"])
        for key, fam, n in (("a", "family_src", r["n"]), ("b", "family_tgt", r["m"])):
            p, p0 = np.asarray(r[key]), np.asarray(r0[key])
            k = int((p == 0).sum())
            assert 1 <= k <= n - 1 or n * cfg.empty_frac_max < 1.5
            assert round(cfg.empty_frac_min * n) <= k <= min(n - 1, round(cfg.empty_frac_max * n))
            np.testing.assert_allclose(p.sum(), 1.0, atol=1e-12)
            kept = p > 0
            np.testing.assert_allclose(p[kept], p0[kept] / p0[kept].sum(), rtol=1e-12)   # the rest is renormalised
            assert r[fam].startswith(cfg.marginals + "+empty_")
            patterns.add(r[fam].split("+empty_")[1])
        P = dense(r)
        np.testing.assert_allclose(P.sum(1), r["a"], atol=1e-7)
        np.testing.assert_allclose(P.sum(0), r["b"], atol=1e-7)
        assert np.all(P[np.asarray(r["a"]) == 0] == 0) and np.all(P[:, np.asarray(r["b"]) == 0] == 0)
    assert patterns == {"random", "blobs", "holes"}


def test_probability_is_per_marginal():
    cfg = GenConfig(dim=2, empty_bins_prob=0.5)
    emptied = [("+empty_" in make_random_instance(s, cfg)[f]) for s in range(400) for f in ("family_src", "family_tgt")]
    assert 0.4 < np.mean(emptied) < 0.6


def test_rejects_bad_settings():
    with pytest.raises(ValueError):
        GenConfig(empty_bins_prob=1.5)
    with pytest.raises(ValueError):
        GenConfig(empty_frac_min=0.5, empty_frac_max=0.2)
