"""2D corpus generation: grid order, random instances, named 2D distributions (OOD split)."""
import numpy as np

from torot.data.families import NAMED_2D_FAMILIES, named_marginal_2d
from torot.data.generate import GenConfig, make_ood_instances, make_random_instance
from torot.data.supports import grid_2d


def test_grid_2d_is_x_major():
    g = grid_2d(3)
    assert g.shape == (9, 2)
    np.testing.assert_allclose(g[0], [1 / 6, 1 / 6])
    np.testing.assert_allclose(g[1], [1 / 6, 3 / 6])          # second index moves along y
    np.testing.assert_allclose(g[3], [3 / 6, 1 / 6])          # then x
    assert (g >= 0).all() and (g <= 1).all()


def test_random_2d_instance_is_a_feasible_square_grid():
    cfg = GenConfig(dim=2)
    for seed in range(20):
        r = make_random_instance(seed, cfg)
        k = int(round(np.sqrt(r["n"])))
        assert r["dim"] == 2 and r["n"] == r["m"] == k * k and cfg.side_min <= k <= cfg.side_max
        assert np.asarray(r["coords_src"]).shape == (r["n"], 2)
        P = np.zeros((r["n"], r["m"]))
        P[r["plan_rows"], r["plan_cols"]] = r["plan_vals"]
        np.testing.assert_allclose(P.sum(1), r["a"], atol=1e-7)
        np.testing.assert_allclose(P.sum(0), r["b"], atol=1e-7)
        assert len(r["plan_vals"]) <= r["n"] + r["m"] - 1


def test_extrapolation_sides():
    cfg = GenConfig(dim=2)
    sides = {int(round(np.sqrt(make_random_instance(s, cfg, extrapolate=True)["n"]))) for s in range(30)}
    assert sides <= set(range(cfg.extrap_side_min, cfg.extrap_side_max + 1))


def test_named_2d_marginals_are_deterministic_histograms():
    for name in NAMED_2D_FAMILIES:
        a, b = named_marginal_2d(name, 6), named_marginal_2d(name, 6)
        assert a.shape == (36,) and np.isclose(a.sum(), 1.0) and (a >= 0).all()
        np.testing.assert_array_equal(a, b)


def test_ood_2d_split_has_all_ordered_pairs():
    recs = list(make_ood_instances(GenConfig(dim=2, ood_sides=[4], cost_types=["sqeuclidean"])))
    assert len(recs) == len(NAMED_2D_FAMILIES) ** 2
    assert {(r["family_src"], r["family_tgt"]) for r in recs} == {(a, b) for a in NAMED_2D_FAMILIES for b in NAMED_2D_FAMILIES}
