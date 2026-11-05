"""Score every prediction in preds/ against the exact plans with one set of metrics (CPU, env_torot).

    python reproduce/score.py experiments/torot/camera_ready          (env_torot, CPU)

Per instance (scores/per_instance/<set>__<method>.csv):
  paper_mape        the paper's MAPE (weighted over the union support; = L1 plan error since the plan sums to 1)
  mae, hellinger, tv, ot_cost_abs_diff   the paper's other plan metrics
                    (MAE over the entries where the plans differ; Hellinger; total variation 0.5*L1; |<C,P> - opt|)
  mape_ref_support  mean relative error over the exact plan's nonzeros
  support_recall / support_precision (every nonzero) / support_precision_1e-4 (entries >= 1e-4)
  row_violation_l1, col_violation_l1, max_violation, cost_gap (signed), cost_gap_rel, nnz_pred, nnz_ref
Summary (scores/summary.json, scores/summary.md): mean, std, median, p10, p90 per metric, plus timings.
Marginal residual profiles (scores/violations/<set>__<method>.npz): |P 1 - a| per row and |P^T 1 - b| per column.
"""
import csv, glob, json, os, sys
import numpy as np
from torot.data.cost import cost_matrix
from torot.data.supports import set_coords

OUT = sys.argv[1]
for sub in ("scores/per_instance", "scores/violations"):
    os.makedirs(f"{OUT}/{sub}", exist_ok=True)
METRICS = ["paper_mape", "mae", "hellinger", "tv", "ot_cost_abs_diff", "mape_ref_support", "support_recall", "support_precision", "support_precision_1e-4",
           "row_violation_l1", "col_violation_l1", "max_violation", "cost_gap", "cost_gap_rel", "nnz_pred", "nnz_ref"]

def paper_mape(A, B, eps=1e-12):
    idx = np.union1d(np.flatnonzero(A), np.flatnonzero(B))
    if idx.size == 0:
        return 0.0
    y_true, y_pred = A.ravel()[idx], B.ravel()[idx]
    y_true = np.where(np.abs(y_true) < eps, np.sign(y_true) * eps, y_true)
    y_true = np.where(y_true == 0.0, eps, y_true)
    return float(np.abs(y_true - y_pred).sum() / np.abs(y_true).sum())

def instance_metrics(P, R, a, b, C, opt):
    ref, pred, pred4 = R > 0, P > 0, P >= 1e-4
    row, col = np.abs(P.sum(1) - a), np.abs(P.sum(0) - b)
    gap = float((P * C).sum() - opt)
    d = R - P
    return {"paper_mape": paper_mape(R, P),
            "mae": float(np.abs(d[d != 0]).mean()) if (d != 0).any() else 0.0,
            "hellinger": float(np.sqrt(((np.sqrt(np.clip(R, 0, None)) - np.sqrt(np.clip(P, 0, None))) ** 2).sum()) / np.sqrt(2.0)),
            "tv": float(0.5 * np.abs(d).sum()),
            "ot_cost_abs_diff": abs(gap),
            "mape_ref_support": float(np.mean(np.abs(P[ref] - R[ref]) / R[ref])),
            "support_recall": float((ref & pred).sum() / ref.sum()),
            "support_precision": float((ref & pred).sum() / max(pred.sum(), 1)),
            "support_precision_1e-4": float((ref & pred4).sum() / max(pred4.sum(), 1)),
            "row_violation_l1": float(row.sum()), "col_violation_l1": float(col.sum()),
            "max_violation": float(max(row.max(), col.max())), "cost_gap": gap,
            "cost_gap_rel": gap / opt if opt > 1e-9 else float("nan"),
            "nnz_pred": int(pred.sum()), "nnz_ref": int(ref.sum())}

summary = {}
for set_path in sorted(glob.glob(f"{OUT}/sets/*.npz")):
    name = os.path.basename(set_path)[:-4]; z = np.load(set_path)
    C = cost_matrix(set_coords(z), set_coords(z), str(z["cost"]))
    for pred_path in sorted(glob.glob(f"{OUT}/preds/{name}__*.npy")):
        method = os.path.basename(pred_path)[len(name) + 2:-4]
        preds = np.load(pred_path)
        rows = [{"src": s, "tgt": t, **instance_metrics(P, R, a, b, C, opt)}
                for P, R, a, b, opt, s, t in zip(preds, z["plan"], z["a"], z["b"], z["ot_cost"], z["src"], z["tgt"])]
        os.makedirs(f"{OUT}/scores/violations", exist_ok=True)
        np.savez(f"{OUT}/scores/violations/{name}__{method}.npz",
                 row=np.abs(preds.sum(2) - z["a"]), col=np.abs(preds.sum(1) - z["b"]))
        with open(f"{OUT}/scores/per_instance/{name}__{method}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        stats = {}
        for m in METRICS:
            v = np.array([r[m] for r in rows], dtype=float); v = v[np.isfinite(v)]
            stats[m] = {"mean": float(v.mean()), "std": float(v.std()), "median": float(np.median(v)),
                        "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}
        tpath = pred_path[:-4] + ".timing.json"
        stats["timing"] = json.load(open(tpath)) if os.path.exists(tpath) else None
        summary.setdefault(name, {})[method] = stats
# POT has no prediction file (it is the reference); keep its timing
for tpath in sorted(glob.glob(f"{OUT}/preds/*__pot.timing.json")):
    name = os.path.basename(tpath)[: -len("__pot.timing.json")]
    summary.setdefault(name, {})["pot"] = {"timing": json.load(open(tpath))}
json.dump(summary, open(f"{OUT}/scores/summary.json", "w"), indent=2)

ORDER = ["pot", "sinkhorn_eps0.01", "sinkhorn_eps0.001", "metaot", "unot", "v2row_lastrow", "v2row_sym_lastrow", "neuralot"]
with open(f"{OUT}/scores/summary.md", "w") as f:
    for name in sorted(summary):
        f.write(f"\n### {name}\n\n| method | paper MAPE (mean ± std) | paper MAPE (median) | recall | precision (≥1e-4) | max violation (median) | cost gap (median) | ms / instance (batched) | s / instance (single) |\n|---|---|---|---|---|---|---|---|---|\n")
        for method in sorted(summary[name], key=lambda m: ORDER.index(m) if m in ORDER else 99):
            s = summary[name][method]; t = s.get("timing") or {}
            tb = f"{t['batched_s_per_instance'] * 1e3:.2f}" if t else "–"; ts = f"{t['single_instance_s_median']:.3g}" if t else "–"
            if "paper_mape" not in s:
                f.write(f"| {method} (exact) | 0 | 0 | 1 | 1 | 0 | 0 | {tb} | {ts} |\n"); continue
            f.write(f"| {method} | {s['paper_mape']['mean']:.4f} ± {s['paper_mape']['std']:.4f} | {s['paper_mape']['median']:.4f} | "
                    f"{s['support_recall']['mean']:.3f} | {s['support_precision_1e-4']['mean']:.3f} | {s['max_violation']['median']:.1e} | "
                    f"{s['cost_gap']['median']:+.1e} | {tb} | {ts} |\n")
print(open(f"{OUT}/scores/summary.md").read())
