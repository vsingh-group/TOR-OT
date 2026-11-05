"""Shared helpers (env_torot or any baseline environment):  pytest baselines/tests/test_common.py"""
import json

import numpy as np
import ot
import pytest

from baselines.common import cost_matrix, paper_mape, plan_summary, select_run


@pytest.mark.parametrize("cost", ["sqeuclidean", "euclidean", "canberra"])
def test_cost_matrix_is_max_normalised_and_symmetric(cost):
    C = cost_matrix(30, cost)
    assert C.shape == (30, 30) and np.isclose(C.max(), 1.0) and np.allclose(C, C.T) and np.allclose(np.diag(C), 0)


def test_paper_mape_is_relative_l1_error():
    rng = np.random.default_rng(0)
    a, b = rng.dirichlet(np.ones(8)), rng.dirichlet(np.ones(8))
    R = ot.emd(a, b, cost_matrix(8, "sqeuclidean"))
    assert paper_mape(R, R) == 0.0
    P = R + 1e-3 * rng.random(R.shape)
    assert np.isclose(paper_mape(R, P), np.abs(P - R).sum() / R.sum())


def test_plan_summary_of_exact_plans_is_zero():
    rng = np.random.default_rng(1)
    a, b = rng.dirichlet(np.ones(6), 4), rng.dirichlet(np.ones(6), 4)
    C = cost_matrix(6, "sqeuclidean")
    plans = np.stack([ot.emd(x, y, C) for x, y in zip(a, b)])
    s = {"name": "t", "a": a, "b": b, "plan": plans, "cost": "sqeuclidean", "n": 6}
    out = plan_summary(plans, s)
    assert out["paper_mape_mean"] == 0.0 and out["abs_cost_gap_mean"] < 1e-12 and out["row_violation_l1_mean"] < 1e-12


def test_select_run_picks_lowest_validation_mape(tmp_path):
    for name, cost, mape in (("x1", "sqeuclidean", 0.3), ("x2", "sqeuclidean", 0.1), ("x3", "canberra", 0.01)):
        (tmp_path / name).mkdir()
        json.dump({"cost": cost, "eps": 0.01}, open(tmp_path / name / "config.json", "w"))
        json.dump({"validation": {"paper_mape_mean": mape}}, open(tmp_path / name / "metrics.json", "w"))
    runs = [str(tmp_path / n) for n in ("x1", "x2", "x3")]
    best, info = select_run(runs, "sqeuclidean")
    assert best.endswith("x2") and set(info["candidates"]) == set(runs[:2])


def test_select_run_refuses_unfinished_candidates(tmp_path):
    for name, has_metrics in (("done", True), ("running", False)):
        (tmp_path / name).mkdir()
        json.dump({"cost": "gaussian", "eps": 0.01}, open(tmp_path / name / "config.json", "w"))
        if has_metrics:
            json.dump({"validation": {"paper_mape_mean": 0.2}}, open(tmp_path / name / "metrics.json", "w"))
    with pytest.raises(RuntimeError):
        select_run([str(tmp_path / "done"), str(tmp_path / "running")], "gaussian")
