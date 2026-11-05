"""Score each method against the solution of the problem it solves (CPU, env_torot). Used by Figure 3.

    python reproduce/score_reference.py experiments/torot/camera_ready [--sets 'id_simplex_*' 'ood_paper_*'] [--workers 12]

score.py compares every prediction with POT's exact plan. Two things make that the wrong reference for some methods:
  * Meta OT and UNOT solve entropic OT at a fixed eps (their plans are diag(u) K diag(v), K = exp(-C / eps)). Their
    target is the entropic plan, not the exact one. Reference: the converged Sinkhorn plan at the same eps
    (preds/<set>__sinkhorn_gpu_eps<eps>.npy, from baselines/pot/sinkhorn_gpu.py), with eps read from each run's timing file.
  * Exact OT can have many optimal plans (in 1D, |x-y| has ties), and POT returns one of them. Reference for methods
    that target exact OT (ToR-OT, NeuralOT): the optimal plan closest to the prediction in L1, from a linear program
    (min ||P - P_hat||_1 over feasible P with <C, P> <= OT cost). It is POT's plan whenever the optimum is unique.

NeuralOT runs on a subset of a large set are read from preds_subset/<set>__neuralot.npz (only those instances are scored).

Per instance (scores/per_instance_ref/<set>__<method>.csv):
  mape_vs_reference   the paper's MAPE (score.py) against the reference above
  mape_vs_pot         the same against POT's plan (= score.py's paper_mape)
  mae / hellinger / tv_vs_reference   score.py's other plan metrics, against the reference (App. E figures)
  ot_cost_diff_vs_reference   |<C, P_hat> - <C, reference>|
  reference, reference_marginal_l1 (how far the reference is from the marginals; Sinkhorn: convergence)
"""
import argparse, csv, fnmatch, glob, json, os
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from torot.data.cost import cost_matrix
from torot.data.supports import set_coords

ENTROPIC_REF = "sinkhorn_gpu_eps{eps:g}"


def paper_mape(A, B, eps=1e-12):
    """The paper's MAPE with A as the reference (identical to reproduce/score.py)."""
    idx = np.union1d(np.flatnonzero(A), np.flatnonzero(B))
    if idx.size == 0:
        return 0.0
    y_true, y_pred = A.ravel()[idx], B.ravel()[idx]
    y_true = np.where(np.abs(y_true) < eps, np.sign(y_true) * eps, y_true)
    y_true = np.where(y_true == 0.0, eps, y_true)
    return float(np.abs(y_true - y_pred).sum() / np.abs(y_true).sum())


def other_metrics(Q, P):
    """score.py's MAE, Hellinger distance and total variation of P against the reference Q."""
    d = Q - P
    return {"mae_vs_reference": float(np.abs(d[d != 0]).mean()) if (d != 0).any() else 0.0,
            "hellinger_vs_reference": float(np.sqrt(((np.sqrt(np.clip(Q, 0, None)) - np.sqrt(np.clip(P, 0, None))) ** 2).sum()) / np.sqrt(2.0)),
            "tv_vs_reference": float(0.5 * np.abs(d).sum())}


def nearest_optimal_plan(P_hat, a, b, C, opt):
    """argmin ||P - P_hat||_1 over transport plans P of (a, b) with <C, P> <= opt (the optimal face), via HiGHS."""
    n, m = C.shape
    N = n * m
    I = sp.identity(N, format="csr")
    rows = sp.kron(sp.identity(n), np.ones((1, m)))              # P 1 = a
    cols = sp.kron(np.ones((1, n)), sp.identity(m))              # P^T 1 = b
    A_eq = sp.hstack([sp.vstack([rows, cols]), sp.csr_matrix((n + m, N))]).tocsr()
    A_ub = sp.vstack([sp.hstack([I, -I]), sp.hstack([-I, -I]),
                      sp.hstack([sp.csr_matrix(C.ravel()), sp.csr_matrix((1, N))])]).tocsr()
    p = P_hat.ravel()
    b_eq = np.concatenate([a, b * (a.sum() / b.sum())])
    b_ub = np.concatenate([p, -p, [opt + 1e-9]])
    c = np.concatenate([np.zeros(N), np.ones(N)])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if res.status == 2:   # HiGHS presolve can wrongly report infeasibility when some masses are tiny (~1e-9): solve without it
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs",
                      options={"presolve": False})
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    return np.clip(res.x[:N].reshape(n, m), 0, None)


def score_exact(job):
    """(set path, prediction path, indices) -> rows, for a method that targets exact OT."""
    set_path, pred_path, ks = job
    z = np.load(set_path)
    C = cost_matrix(set_coords(z), set_coords(z), str(z["cost"]))
    if pred_path.endswith(".npz"):                                   # subset run (NeuralOT): plans of the instances idx
        zz = np.load(pred_path)
        row = {int(k): r for r, k in enumerate(zz["idx"])}
        preds, ks_pos = zz["plans"], [row[int(k)] for k in ks]
    else:
        preds, ks_pos = np.load(pred_path, mmap_mode="r"), ks
    out = []
    for k, r in zip(ks, ks_pos):
        P, R, a, b = np.asarray(preds[r], dtype=np.float64), z["plan"][k], z["a"][k], z["b"][k]
        opt = float((C * R).sum())
        Q = nearest_optimal_plan(P, a, b, C, opt)
        out.append({"index": int(k), "reference": "exact: nearest optimal plan",
                    "mape_vs_reference": paper_mape(Q, P), "mape_vs_pot": paper_mape(R, P),
                    "ot_cost_diff_vs_reference": abs(float((C * P).sum()) - opt), "reference_cost": opt,
                    "reference_marginal_l1": float(np.abs(Q.sum(1) - a).sum() + np.abs(Q.sum(0) - b).sum()),
                    "reference_vs_pot_l1": float(np.abs(Q - R).sum()), **other_metrics(Q, P)})
    return out


def score_entropic(set_path, pred_path, ref_path, eps):
    z = np.load(set_path)
    C = cost_matrix(set_coords(z), set_coords(z), str(z["cost"]))
    preds, refs = np.load(pred_path), np.load(ref_path)
    out = []
    for k, (P, Q) in enumerate(zip(preds, refs)):
        a, b = z["a"][k], z["b"][k]
        out.append({"index": k, "reference": f"entropic: Sinkhorn eps={eps:g}",
                    "mape_vs_reference": paper_mape(Q, P), "mape_vs_pot": paper_mape(z["plan"][k], P),
                    "ot_cost_diff_vs_reference": abs(float((C * P).sum() - (C * Q).sum())), "reference_cost": float((C * Q).sum()),
                    "reference_marginal_l1": float(np.abs(Q.sum(1) - a).sum() + np.abs(Q.sum(0) - b).sum()),
                    "reference_vs_pot_l1": float(np.abs(Q - z["plan"][k]).sum()), **other_metrics(Q, P)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root")
    ap.add_argument("--sets", nargs="+", default=["id_simplex_*", "ood_paper_*"])
    ap.add_argument("--methods", nargs="+", default=["v2row_lastrow", "neuralot", "metaot", "unot",
                                                     "metaot_seed1", "metaot_seed2", "unot_seed1", "unot_seed2"])
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    out_dir = f"{args.root}/scores/per_instance_ref"
    os.makedirs(out_dir, exist_ok=True)
    sets = sorted(p for p in glob.glob(f"{args.root}/sets/*.npz")
                  if any(fnmatch.fnmatch(os.path.basename(p)[:-4], pat) for pat in args.sets))
    exact_jobs, exact_keys = [], []
    for set_path in sets:
        name = os.path.basename(set_path)[:-4]
        for method in args.methods:
            pred_path = f"{args.root}/preds/{name}__{method}.npy"
            subset = f"{args.root}/preds_subset/{name}__{method}.npz"
            if not os.path.exists(pred_path):
                if os.path.exists(subset):                          # NeuralOT on a subset of the instances
                    idx = np.load(subset)["idx"]
                    for ks in np.array_split(idx, max(1, len(idx) // 25)):
                        exact_jobs.append((set_path, subset, ks)); exact_keys.append((name, method))
                continue
            tpath = pred_path[:-4] + ".timing.json"
            eps = json.load(open(tpath)).get("eps") if os.path.exists(tpath) else None
            if eps is not None:                                     # entropic method: Sinkhorn reference at its eps
                ref_path = f"{args.root}/preds/{name}__{ENTROPIC_REF.format(eps=eps)}.npy"
                write(out_dir, name, method, score_entropic(set_path, pred_path, ref_path, eps))
            else:                                                   # exact method: nearest optimal plan (LPs, in parallel)
                N = len(np.load(pred_path, mmap_mode="r"))
                for ks in np.array_split(np.arange(N), max(1, N // 25)):
                    exact_jobs.append((set_path, pred_path, ks)); exact_keys.append((name, method))
    print(f"{len(exact_jobs)} LP batches on {args.workers} workers", flush=True)
    results = {}
    with Pool(args.workers) as pool:
        for key, rows in zip(exact_keys, pool.imap(score_exact, exact_jobs)):
            results.setdefault(key, []).extend(rows)
    for (name, method), rows in results.items():
        write(out_dir, name, method, sorted(rows, key=lambda r: r["index"]))


def write(out_dir, name, method, rows):
    with open(f"{out_dir}/{name}__{method}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    mv, mp = np.mean([r["mape_vs_reference"] for r in rows]), np.mean([r["mape_vs_pot"] for r in rows])
    print(f"{name:28s} {method:15s} {rows[0]['reference']:30s} MAPE vs reference {mv:.4f}  (vs POT {mp:.4f})", flush=True)


if __name__ == "__main__":
    main()
