"""Evaluation metrics on dense plans.

Solution-invariant metrics (do not depend on which optimal vertex the solver returned):
    cost_gap, row/col marginal violation (L1), max violation (L-inf).
Plan-level metrics (compare against the reference plan):
    MAPE on the reference support, weighted MAPE, support precision / recall.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from torot.data.cost import COST_TYPES, cost_matrix


def marginal_violations(P: np.ndarray, a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    row_err = np.abs(P.sum(1) - a)
    col_err = np.abs(P.sum(0) - b)
    return {
        "row_violation_l1": float(row_err.sum()),
        "col_violation_l1": float(col_err.sum()),
        "max_violation": float(max(row_err.max(), col_err.max())),
    }


def cost_gap(P: np.ndarray, C: np.ndarray, opt_cost: float) -> Dict[str, float]:
    """Signed gap <C, P> - opt. Negative values are possible for infeasible plans."""
    cost = float((P * C).sum())
    return {"cost": cost, "cost_gap": cost - opt_cost, "cost_gap_rel": (cost - opt_cost) / max(opt_cost, 1e-12)}


def plan_errors(P_pred: np.ndarray, P_ref: np.ndarray, support_tol: float = 1e-12) -> Dict[str, float]:
    ref_support = P_ref > support_tol
    pred_support = P_pred > support_tol
    out: Dict[str, float] = {}
    if ref_support.any():
        rel = np.abs(P_pred[ref_support] - P_ref[ref_support]) / P_ref[ref_support]
        out["mape_ref_support"] = float(rel.mean())
        out["wmape"] = float(np.abs(P_pred - P_ref).sum() / P_ref.sum())
        tp = float((ref_support & pred_support).sum())
        out["support_recall"] = tp / float(ref_support.sum())
        out["support_precision"] = tp / float(pred_support.sum()) if pred_support.any() else 0.0
    out["mae"] = float(np.abs(P_pred - P_ref).mean())
    out["false_positive_mass"] = float(P_pred[~ref_support].sum())
    return out


def evaluate_instance(P_pred: np.ndarray, inst: dict) -> Dict[str, float]:
    """All metrics for one instance (``inst`` as produced by ``record_to_instance``)."""
    n, m = inst["n"], inst["m"]
    dim = inst["dim"]
    C = cost_matrix(inst["coords_src"][:n, :dim], inst["coords_tgt"][:m, :dim], COST_TYPES[inst["cost_type"]])
    P_ref = np.zeros((n, m))
    P_ref[inst["plan_rows"], inst["plan_cols"]] = inst["plan_vals"]
    out = {}
    out.update(marginal_violations(P_pred, inst["a"], inst["b"]))
    out.update(cost_gap(P_pred, C, inst["ot_cost"]))
    out.update(plan_errors(P_pred, P_ref))
    out["nnz_pred"] = float((P_pred > 0).sum())
    out["nnz_ref"] = float(len(inst["plan_vals"]))
    return out


def aggregate(records: list[Dict[str, float]]) -> Dict[str, float]:
    """Mean and median of every metric across instances."""
    if not records:
        return {}
    keys = sorted({k for r in records for k in r})
    out = {}
    for k in keys:
        vals = np.asarray([r[k] for r in records if k in r], dtype=np.float64)
        out[f"{k}/mean"] = float(vals.mean())
        out[f"{k}/median"] = float(np.median(vals))
    return out
